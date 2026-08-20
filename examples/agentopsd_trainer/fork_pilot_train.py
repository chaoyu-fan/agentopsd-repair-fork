"""Standalone paired LoRA pilot runner for the asset-free ``fsm-code-v1`` task.

This is engineering infrastructure, not a paper-reproduction script. It loads
one local base checkpoint per arm, adds a small LoRA adapter, collects four
three-turn rollouts, and performs a short PPO/GRPO-style update. The AgentOPSD
arm delegates credit assignment to the upstream ``opsd_utils`` functions; it
does not reimplement their formulas.

The module intentionally does not import Ray, FSDP, vLLM, or any model package
at module import time. Runtime model imports happen only after the CLI validates
that ``--model-path`` is an existing local directory. This keeps CPU/static
validation asset-free and prevents accidental downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from examples.agentopsd_trainer.fork_pilot_fsm import (
        ARMS,
        GROUP_SIZE,
        MAX_NEW_TOKENS,
        MAX_TURNS,
        PILOT_LABEL,
        TASK_NAME,
        FSMCodeEnvironment,
        FSMCodeTask,
        generate_canonical_splits,
        oracle_skill_view,
        schema_hash,
        split_hashes,
        student_view,
    )
except ModuleNotFoundError as exc:
    # Direct execution puts examples/agentopsd_trainer, not the repo root, on
    # sys.path. Add only this fork's root; never search or install packages.
    if exc.name not in ("examples", "examples.agentopsd_trainer"):
        raise
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.agentopsd_trainer.fork_pilot_fsm import (
        ARMS,
        GROUP_SIZE,
        MAX_NEW_TOKENS,
        MAX_TURNS,
        PILOT_LABEL,
        TASK_NAME,
        FSMCodeEnvironment,
        FSMCodeTask,
        generate_canonical_splits,
        oracle_skill_view,
        schema_hash,
        split_hashes,
        student_view,
    )


DEFAULT_MODEL_PATH = "/work/FCY/.agentopsd-pilot-cache/Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_OUTPUT_DIR = "outputs/fork_engineering_pilot"
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_GRAD_NORM = 1.0
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
DETERMINISM_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
V0_STRATEGY = "group_mean"
DETERMINISM_POLICY = {
    "cublas_workspace_config": DETERMINISM_ENVIRONMENT["CUBLAS_WORKSPACE_CONFIG"],
    "torch_use_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False,
}


@dataclass(frozen=True)
class RunnerConfig:
    """CLI-visible settings; task shape and arm names remain fixed by the pilot."""

    model_path: str = DEFAULT_MODEL_PATH
    base_revision: str = DEFAULT_BASE_REVISION
    output_dir: str = DEFAULT_OUTPUT_DIR
    arm: str = "both"
    updates: int = 1
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bf16"
    learning_rate: float = DEFAULT_LEARNING_RATE
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    opsd_granularity: str = "turn"

    def __post_init__(self) -> None:
        if self.arm not in ("both", *ARMS):
            raise ValueError("arm must be both, grpo, or agentopsd")
        if not self.base_revision:
            raise ValueError("base_revision must be non-empty")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if self.dtype not in ("bf16", "fp32"):
            raise ValueError("dtype must be bf16 or fp32")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be in (0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.lora_dropout < 0.0 or self.lora_dropout >= 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.opsd_granularity not in ("turn", "token"):
            raise ValueError("opsd_granularity must be turn or token")

    @property
    def arms(self) -> Tuple[str, ...]:
        return ARMS if self.arm == "both" else (self.arm,)

    def manifest(self) -> Dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "label": PILOT_LABEL,
                "task_name": TASK_NAME,
                "max_turns": MAX_TURNS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "group_size": GROUP_SIZE,
                "train_size": 8,
                "eval_size": 4,
                "schema_sha256": schema_hash(),
                "v0_strategy": V0_STRATEGY,
                "determinism_policy": DETERMINISM_POLICY,
            }
        )
        return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _enforce_offline() -> Dict[str, Any]:
    """Disable Hub access before any runtime model package is imported."""
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value
    return {**OFFLINE_ENVIRONMENT, "local_files_only": True}


def _enforce_determinism_environment() -> Dict[str, str]:
    for name, value in DETERMINISM_ENVIRONMENT.items():
        os.environ[name] = value
    return dict(DETERMINISM_ENVIRONMENT)


class PairedRolloutMismatch(RuntimeError):
    """Raised when common-randomness token outputs differ before the first update."""


def _configure_determinism(torch: Any) -> Dict[str, Any]:
    """Enable deterministic kernels or fail closed before model execution."""
    _enforce_determinism_environment()
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return dict(DETERMINISM_POLICY)


def _sampling_seed(
    config: RunnerConfig,
    task: FSMCodeTask,
    phase: str,
    phase_index: int,
    rollout_index: int,
    turn: int,
) -> int:
    """Derive an arm-independent 63-bit seed for one generation call."""
    value = {
        "base_revision": config.base_revision,
        "group_size": GROUP_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_turns": MAX_TURNS,
        "phase": phase,
        "phase_index": phase_index,
        "rollout_index": rollout_index,
        "seed": config.seed,
        "task_hash": task.task_hash,
        "task_id": task.task_id,
        "turn": turn,
    }
    return int(_canonical_hash(value)[:16], 16) % (2**63 - 1)


def _set_sampling_seed(torch: Any, device: Any, seed: int) -> None:
    """Set the exact RNG used immediately before model.generate()."""
    torch.manual_seed(seed)
    if getattr(device, "type", None) == "cuda":
        torch.cuda.manual_seed(seed)


def _completion_fingerprint(completion_ids: Sequence[int]) -> str:
    return _canonical_hash([int(token_id) for token_id in completion_ids])


def _rollout_output_fingerprints(
    records: Sequence["TurnRecord"],
) -> Dict[int, str]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record.rollout_index, []).append(
            {
                "completion_fingerprint": record.completion_fingerprint,
                "turn": record.turn,
            }
        )
    return {
        rollout_index: _canonical_hash(sorted(turns, key=lambda value: value["turn"]))
        for rollout_index, turns in grouped.items()
    }


def _compare_paired_fingerprints(
    reference: Dict[int, str],
    candidate: Dict[int, str],
) -> None:
    if reference != candidate:
        raise PairedRolloutMismatch(
            "pre-update rollout token fingerprints differ: reference={} candidate={}".format(
                reference, candidate
            )
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_files_sha256(root: Path, patterns: Sequence[str]) -> str:
    files = sorted(
        path
        for pattern in patterns
        for path in root.rglob(pattern)
        if path.is_file() and not path.is_symlink()
    )
    entries = [
        {"path": str(path.relative_to(root)), "sha256": _file_sha256(path)}
        for path in files
    ]
    return _canonical_hash(entries)


def _git_commit(repo_root: Path) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(repo_root),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return "{}-dirty".format(head) if dirty else head


def _provenance(config: RunnerConfig) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    model_root = Path(config.model_path).resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(
            "model path must be an existing local directory; downloads are disabled: {}".format(
                model_root
            )
        )
    split_digest = split_hashes(seed=config.seed)
    paired_config = asdict(config)
    paired_config["arm"] = "both"
    source_hashes = {
        "fsm_sha256": _file_sha256(repo_root / "examples/agentopsd_trainer/fork_pilot_fsm.py"),
        "opsd_utils_sha256": _file_sha256(repo_root / "verl/trainer/ppo/opsd_utils.py"),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
    }
    return {
        "git_commit": _git_commit(repo_root),
        "source_hashes": source_hashes,
        "source_aggregate_sha256": _canonical_hash(source_hashes),
        "model_path": str(model_root),
        "model_revision": config.base_revision,
        "model_aggregate_sha256": _aggregate_files_sha256(
            model_root, ("*.safetensors", "*.bin", "*.pt")
        ),
        "tokenizer_metadata_sha256": _aggregate_files_sha256(
            model_root,
            (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
                "spiece.model",
            ),
        ),
        "config_metadata_sha256": _aggregate_files_sha256(
            model_root, ("config.json", "generation_config.json")
        ),
        "split_hashes": split_digest,
        "schema_sha256": schema_hash(),
        "v0_strategy": V0_STRATEGY,
        "determinism_policy": dict(DETERMINISM_POLICY),
        "paired_config_sha256": _canonical_hash(paired_config),
        "paired_arm_invariants": {
            "base_revision": config.base_revision,
            "model_path": str(model_root),
            "seed": config.seed,
            "shared_config_sha256": _canonical_hash(paired_config),
        },
        "offline": _enforce_offline(),
    }


def _rollout_spec(
    config: RunnerConfig,
    task: FSMCodeTask,
    phase: str,
    phase_index: int,
) -> Dict[str, Any]:
    """Describe one arm-independent seeded rollout condition for audit logs."""
    if phase not in ("train", "eval"):
        raise ValueError("phase must be train or eval")
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_turns": MAX_TURNS,
        "phase": phase,
        "phase_index": phase_index,
        "sampling": {
            "temperature": config.temperature,
            "top_p": config.top_p,
        },
        "seed": config.seed,
        "task_hash": task.task_hash,
        "task_id": task.task_id,
        "task_split": task.split,
        "group_size": GROUP_SIZE,
    }


def _rollout_sha256(
    config: RunnerConfig,
    task: FSMCodeTask,
    phase: str,
    phase_index: int,
) -> str:
    return _canonical_hash(_rollout_spec(config, task, phase, phase_index))


def _rollout_plan_sha256(
    config: RunnerConfig,
    tasks: Dict[str, Tuple[FSMCodeTask, ...]],
) -> str:
    specs = [
        _rollout_spec(
            config,
            tasks["train"][update % len(tasks["train"])],
            "train",
            update,
        )
        for update in range(config.updates)
    ]
    specs.extend(
        _rollout_spec(config, task, "eval", eval_index)
        for eval_index, task in enumerate(tasks["eval"])
    )
    return _canonical_hash(specs)


def _manifest_payload(
    config: RunnerConfig,
    tasks: Dict[str, Tuple[FSMCodeTask, ...]],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the immutable paired-run manifest before loading either arm."""
    return {
        **config.manifest(),
        **provenance,
        "failure_category": None,
        "rollout_plan_sha256": _rollout_plan_sha256(config, tasks),
    }


def _record_provenance(
    config: RunnerConfig,
    task: FSMCodeTask,
    phase: str,
    phase_index: int,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach the same reproducibility facts to every JSONL observation."""
    return {
        **provenance,
        "failure_category": None,
        "rollout_sha256": _rollout_sha256(config, task, phase, phase_index),
    }


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _failure_category(exc: BaseException, phase: str) -> str:
    if isinstance(exc, PairedRolloutMismatch):
        return "pairing_mismatch"
    if phase == "load":
        return "model_load"
    return "runtime"


def _terminal_record(
    arm: str,
    status: str,
    phase: str,
    update: Optional[int],
    started: float,
    progress: Dict[str, Any],
    exc: Optional[BaseException] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "arm": arm,
        "status": status,
        "phase": phase,
        "update": update,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "peak_vram_mib": progress.get("peak_vram_mib"),
        "failure_category": None if exc is None else _failure_category(exc, phase),
        "exception_type": None if exc is None else type(exc).__name__,
        "exception_message": None if exc is None else str(exc),
    }
    return record


@dataclass
class TurnRecord:
    task_id: str
    split: str
    rollout_index: int
    turn: int
    state_before: str
    state_after: str
    prompt_ids: List[int]
    teacher_prompt_ids: List[int]
    completion_ids: List[int]
    action_text: str
    reward: float
    episode_reward: float
    parsed_action: Optional[str]
    parse_error: Optional[str]
    sampling_seed: int
    completion_fingerprint: str
    student_log_probs: List[float]
    teacher_log_probs: List[float]


@dataclass
class ArmRuntime:
    torch: Any
    model: Any
    tokenizer: Any
    optimizer: Any
    device: Any


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _runtime_dtype(config: RunnerConfig, torch: Any) -> Any:
    if config.dtype == "fp32":
        return torch.float32
    return torch.bfloat16


def _load_runtime(config: RunnerConfig, seed: int) -> ArmRuntime:
    """Load only local assets and create a fresh LoRA model for one arm."""
    _enforce_offline()
    model_path = Path(config.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(
            "model path must be an existing local directory; downloads are disabled: {}".format(
                model_path
            )
        )

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    device = torch.device(config.device)
    _configure_determinism(torch)
    _set_seed(seed, torch)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        revision=config.base_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        revision=config.base_revision,
        trust_remote_code=False,
        torch_dtype=_runtime_dtype(config, torch),
        low_cpu_mem_usage=True,
    )
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    model = get_peft_model(model, lora_config).to(device)
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    print(
        "[{}] loaded arm with seed={} model={} trainable_params={}".format(
            PILOT_LABEL,
            seed,
            model_path,
            sum(parameter.numel() for parameter in trainable),
        )
    )
    return ArmRuntime(torch=torch, model=model, tokenizer=tokenizer, optimizer=optimizer, device=device)


def _tokenize(tokenizer: Any, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]


def _generate_action(
    runtime: ArmRuntime,
    prompt: str,
    config: RunnerConfig,
    sampling_seed: int,
) -> Tuple[List[int], List[int], str]:
    tokenizer = runtime.tokenizer
    prompt_ids = _tokenize(tokenizer, prompt)
    input_ids = runtime.torch.tensor([prompt_ids], dtype=runtime.torch.long, device=runtime.device)
    attention_mask = runtime.torch.ones_like(input_ids)
    _set_sampling_seed(runtime.torch, runtime.device, sampling_seed)
    with runtime.torch.no_grad():
        generated = runtime.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    completion_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
    action_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return prompt_ids, completion_ids, action_text


def _score_completion(
    runtime: ArmRuntime,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    requires_grad: bool,
) -> Any:
    """Return one log-probability per completion token from exact token IDs."""
    torch = runtime.torch
    if not completion_ids:
        return torch.empty(0, dtype=torch.float32, device=runtime.device)
    input_ids = torch.tensor(
        [list(prompt_ids) + list(completion_ids)], dtype=torch.long, device=runtime.device
    )
    attention_mask = torch.ones_like(input_ids)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        logits = runtime.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits[0]
        prompt_length = len(prompt_ids)
        next_token_logits = logits[prompt_length - 1 : -1]
        log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)
        target = torch.tensor(list(completion_ids), dtype=torch.long, device=runtime.device)
        token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def _teacher_prompt(task: FSMCodeTask, state: str, turn: int) -> str:
    return "{}\n\nPrivileged skill view:\n{}".format(
        student_view(task, state, turn), oracle_skill_view(task, state)
    )


def _collect_group(
    runtime: ArmRuntime,
    task: FSMCodeTask,
    split: str,
    config: RunnerConfig,
    update: int,
) -> Tuple[List[TurnRecord], Dict[str, Any]]:
    """Collect four trajectories and old student/teacher log-prob telemetry."""
    start = time.perf_counter()
    environments = [FSMCodeEnvironment(task) for _ in range(GROUP_SIZE)]
    records: List[TurnRecord] = []
    runtime.model.eval()

    while True:
        active = [
            index for index, environment in enumerate(environments) if not environment.terminated
        ]
        if not active:
            break
        for index in active:
            environment = environments[index]
            state_before = environment.state
            turn = environment.turn
            prompt = student_view(task, state_before, turn)
            teacher_prompt = _teacher_prompt(task, state_before, turn)
            sampling_seed = _sampling_seed(
                config, task, split, update, index, turn
            )
            prompt_ids, completion_ids, action_text = _generate_action(
                runtime, prompt, config, sampling_seed
            )
            teacher_prompt_ids = _tokenize(runtime.tokenizer, teacher_prompt)
            student_lps = _score_completion(runtime, prompt_ids, completion_ids, requires_grad=False)
            teacher_lps = _score_completion(
                runtime, teacher_prompt_ids, completion_ids, requires_grad=False
            )
            step = environment.step(action_text)
            records.append(
                TurnRecord(
                    task_id=task.task_id,
                    split=split,
                    rollout_index=index,
                    turn=turn,
                    state_before=state_before,
                    state_after=step.state_after,
                    prompt_ids=prompt_ids,
                    teacher_prompt_ids=teacher_prompt_ids,
                    completion_ids=completion_ids,
                    action_text=action_text,
                    reward=step.reward,
                    episode_reward=0.0,
                    parsed_action=step.parsed_action,
                    parse_error=step.parse_error,
                    sampling_seed=sampling_seed,
                    completion_fingerprint=_completion_fingerprint(completion_ids),
                    student_log_probs=student_lps.float().cpu().tolist(),
                    teacher_log_probs=teacher_lps.float().cpu().tolist(),
                )
            )

    episode_rewards = {
        index: 1.0 if environment.state == task.goal_state else 0.0
        for index, environment in enumerate(environments)
    }
    for record in records:
        record.episode_reward = episode_rewards[record.rollout_index]
    invalids = sum(record.parse_error is not None for record in records)
    rewards = [episode_rewards[index] for index in range(GROUP_SIZE)]
    output_fingerprints = _rollout_output_fingerprints(records)
    return records, {
        "update": update,
        "task_id": task.task_id,
        "split": split,
        "rewards": rewards,
        "invalid_count": invalids,
        "action_count": len(records),
        "output_fingerprint": _canonical_hash(output_fingerprints),
        "rollout_output_fingerprints": output_fingerprints,
        "rollout_seconds": time.perf_counter() - start,
    }


def _records_to_tensors(
    records: Sequence[TurnRecord],
    torch: Any,
    device: Any,
) -> Tuple[Any, Any, Any]:
    """Pad per-turn log-prob telemetry into ``(rows, max_tokens)`` tensors."""
    max_length = max((len(record.completion_ids) for record in records), default=0)
    shape = (len(records), max_length)
    student = torch.zeros(shape, dtype=torch.float32, device=device)
    teacher = torch.zeros(shape, dtype=torch.float32, device=device)
    mask = torch.zeros(shape, dtype=torch.float32, device=device)
    for row, record in enumerate(records):
        length = len(record.completion_ids)
        if not length:
            continue
        student[row, :length] = torch.tensor(record.student_log_probs, device=device)
        teacher[row, :length] = torch.tensor(record.teacher_log_probs, device=device)
        mask[row, :length] = 1.0
    return student, teacher, mask


def _sequence_advantages(records: Sequence[TurnRecord], torch: Any) -> Any:
    returns_by_rollout: Dict[int, float] = {}
    for record in records:
        existing = returns_by_rollout.setdefault(record.rollout_index, record.episode_reward)
        if existing != record.episode_reward:
            raise ValueError("each rollout must have one episode reward")
    if sorted(returns_by_rollout) != list(range(GROUP_SIZE)):
        raise ValueError("expected exactly {} rollout indices".format(GROUP_SIZE))
    rollout_returns = torch.tensor(
        [returns_by_rollout[index] for index in range(GROUP_SIZE)], dtype=torch.float32
    )
    mean = rollout_returns.mean()
    std = rollout_returns.std(unbiased=False)
    values = ((rollout_returns - mean) / (std + 1e-6)).tolist()
    return torch.tensor(
        [values[record.rollout_index] for record in records], dtype=torch.float32
    )


def _compute_advantages(
    records: Sequence[TurnRecord],
    config: RunnerConfig,
    runtime: ArmRuntime,
) -> Tuple[Any, Dict[str, float]]:
    """Compute fixed GRPO or upstream OPSD advantages for the collected rows."""
    torch = runtime.torch
    student, teacher, mask = _records_to_tensors(records, torch, runtime.device)
    seq_advantage = _sequence_advantages(records, torch).to(runtime.device)
    uid = ["pilot-group"] * len(records)
    traj_uid = ["traj-{}".format(record.rollout_index) for record in records]
    turn_step = [record.turn for record in records]
    rewards = torch.zeros_like(student)
    v0_rewards = torch.zeros_like(student)
    for row, record in enumerate(records):
        if record.reward and mask[row].sum() > 0:
            rewards[row, int(mask[row].sum().item()) - 1] = record.reward
        if record.episode_reward and mask[row].sum() > 0:
            v0_rewards[row, int(mask[row].sum().item()) - 1] = record.episode_reward
    v0_per_traj = None

    if config.arm == "agentopsd":
        from verl.trainer.ppo.opsd_utils import (
            compute_group_mean_v0,
            compute_opsd_token_advantage,
            compute_opsd_turn_advantage,
        )

        v0_per_traj = compute_group_mean_v0(
            # This utility deduplicates by the first row of each trajectory.
            # Each FSM row is a turn, so supply the terminal episode outcome
            # on every turn row solely for the group prior V_0 calculation.
            token_level_rewards=v0_rewards,
            response_mask=mask,
            index=uid,
            traj_index=traj_uid,
        )
        if config.opsd_granularity == "turn":
            advantages = compute_opsd_turn_advantage(
                token_level_rewards=rewards,
                student_log_probs=student,
                teacher_log_probs=teacher,
                response_mask=mask,
                traj_uid=traj_uid,
                turn_step=turn_step,
                episode_rewards=[record.episode_reward for record in records],
                uid=uid,
                v0_per_traj=v0_per_traj,
            )
        else:
            advantages = compute_opsd_token_advantage(
                token_level_rewards=rewards,
                student_log_probs=student,
                teacher_log_probs=teacher,
                response_mask=mask,
                v0_per_traj=v0_per_traj,
            )
    else:
        advantages = seq_advantage.unsqueeze(-1) * mask

    evidence = (teacher - student) * mask
    valid = mask.bool()
    summary = {
        "student_logprob_mean": float(student[valid].mean().item()) if valid.any() else 0.0,
        "teacher_logprob_mean": float(teacher[valid].mean().item()) if valid.any() else 0.0,
        "evidence_mean": float(evidence[valid].mean().item()) if valid.any() else 0.0,
        "evidence_std": float(evidence[valid].std(unbiased=False).item()) if valid.any() else 0.0,
        "advantage_mean": float(advantages[valid].mean().item()) if valid.any() else 0.0,
        "advantage_std": float(advantages[valid].std(unbiased=False).item()) if valid.any() else 0.0,
        "advantage_min": float(advantages[valid].min().item()) if valid.any() else 0.0,
        "advantage_max": float(advantages[valid].max().item()) if valid.any() else 0.0,
    }
    if v0_per_traj is not None:
        summary["v0_mean"] = float(v0_per_traj.float().mean().item())
    return advantages.detach(), summary


def _apply_update(
    records: Sequence[TurnRecord],
    advantages: Any,
    runtime: ArmRuntime,
    config: RunnerConfig,
) -> Dict[str, float]:
    """Apply a clipped policy-gradient update using fresh differentiable scores."""
    torch = runtime.torch
    runtime.model.train()
    runtime.optimizer.zero_grad(set_to_none=True)
    valid_tokens = sum(len(record.completion_ids) for record in records)
    if valid_tokens == 0:
        return {"policy_loss": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0, "updated": 0.0}

    loss_value = 0.0
    clipped = 0
    seen = 0
    scale = float(valid_tokens)
    for row, record in enumerate(records):
        if not record.completion_ids:
            continue
        current = _score_completion(
            runtime, record.prompt_ids, record.completion_ids, requires_grad=True
        )
        old = torch.tensor(record.student_log_probs, dtype=current.dtype, device=runtime.device)
        ratio = torch.exp(current - old)
        adv = advantages[row, : len(record.completion_ids)].to(current.dtype)
        unclipped = ratio * adv
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        clipped_objective = clipped_ratio * adv
        loss = -torch.minimum(unclipped, clipped_objective).sum() / scale
        loss.backward()
        loss_value += float(loss.detach().item())
        clipped += int((ratio.detach() != clipped_ratio.detach()).sum().item())
        seen += len(record.completion_ids)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in runtime.model.parameters() if parameter.requires_grad],
        config.max_grad_norm,
    )
    runtime.optimizer.step()
    return {
        "policy_loss": loss_value,
        "clip_fraction": float(clipped / max(seen, 1)),
        "grad_norm": float(grad_norm.detach().item()),
        "updated": 1.0,
    }


def _peak_vram_mib(torch: Any, device: Any) -> Optional[float]:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 3)


def _jsonable_records(records: Sequence[TurnRecord]) -> List[Dict[str, Any]]:
    return [asdict(record) for record in records]


def _run_arm(
    config: RunnerConfig,
    arm: str,
    tasks: Dict[str, Tuple[FSMCodeTask, ...]],
    provenance: Dict[str, Any],
    pairing_state: Dict[str, Any],
    progress: Dict[str, Any],
) -> None:
    arm_config = RunnerConfig(**{**asdict(config), "arm": arm})
    arm_dir = Path(config.output_dir) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = arm_dir / "telemetry.jsonl"
    arm_seed = config.seed
    progress.update({"phase": "load", "update": None, "peak_vram_mib": None})
    runtime = _load_runtime(arm_config, arm_seed)
    torch = runtime.torch
    with telemetry_path.open("w", encoding="utf-8") as telemetry_file:
        for update in range(config.updates):
            progress.update({"phase": "train", "update": update})
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            total_started = time.perf_counter()
            task = tasks["train"][update % len(tasks["train"])]
            records, rollout_metrics = _collect_group(
                runtime, task, "train", arm_config, update
            )
            progress["peak_vram_mib"] = _peak_vram_mib(torch, runtime.device)
            pairing = {
                "checked": False,
                "scope": "initial_train_pre_update",
                "status": "not_checked",
            }
            if update == 0:
                output_fingerprints = rollout_metrics["rollout_output_fingerprints"]
                if arm == ARMS[0]:
                    pairing_state["reference"] = output_fingerprints
                    pairing_state["status"] = "reference_recorded"
                    pairing["pre_update_rollout_fingerprints"] = output_fingerprints
                    pairing["status"] = "reference_recorded"
                else:
                    pairing_state["candidate"] = output_fingerprints
                    try:
                        _compare_paired_fingerprints(
                            pairing_state.get("reference", {}),
                            output_fingerprints,
                        )
                    except PairedRolloutMismatch:
                        pairing_state["status"] = "mismatch"
                        raise
                    pairing_state["checked"] = True
                    pairing_state["status"] = "matched"
                    pairing["checked"] = True
                    pairing["pre_update_rollout_fingerprints"] = output_fingerprints
                    pairing["status"] = "matched"
            credit_started = time.perf_counter()
            advantages, advantage_summary = _compute_advantages(records, arm_config, runtime)
            credit_seconds = time.perf_counter() - credit_started
            update_started = time.perf_counter()
            update_metrics = _apply_update(records, advantages, runtime, arm_config)
            update_metrics["update_seconds"] = time.perf_counter() - update_started
            update_metrics["total_seconds"] = time.perf_counter() - total_started
            rollout_metrics["credit_seconds"] = credit_seconds
            payload = {
                **_record_provenance(config, task, "train", update, provenance),
                "label": PILOT_LABEL,
                "phase": "train",
                "arm": arm,
                "seed": arm_seed,
                "config": arm_config.manifest(),
                "rollout": rollout_metrics,
                "pairing": pairing,
                "telemetry": {
                    "records": _jsonable_records(records),
                    "advantages": advantage_summary,
                    "update": update_metrics,
                    "peak_vram_mib": _peak_vram_mib(torch, runtime.device),
                },
            }
            telemetry_file.write(json.dumps(payload, sort_keys=True) + "\n")
            telemetry_file.flush()
            print(
                "[{}] arm={} phase=train update={} rewards={} invalids={}".format(
                    PILOT_LABEL,
                    arm,
                    update,
                    rollout_metrics["rewards"],
                    rollout_metrics["invalid_count"],
                )
            )

        for eval_index, task in enumerate(tasks["eval"]):
            progress.update({"phase": "eval", "update": eval_index})
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            total_started = time.perf_counter()
            records, rollout_metrics = _collect_group(
                runtime, task, "eval", arm_config, eval_index
            )
            credit_started = time.perf_counter()
            _, advantage_summary = _compute_advantages(records, arm_config, runtime)
            rollout_metrics["credit_seconds"] = time.perf_counter() - credit_started
            rollout_metrics["total_seconds"] = time.perf_counter() - total_started
            progress["peak_vram_mib"] = _peak_vram_mib(torch, runtime.device)
            payload = {
                **_record_provenance(config, task, "eval", eval_index, provenance),
                "label": PILOT_LABEL,
                "phase": "eval",
                "arm": arm,
                "seed": arm_seed,
                "config": arm_config.manifest(),
                "rollout": rollout_metrics,
                "pairing": {
                    "checked": False,
                    "scope": "initial_train_pre_update",
                    "status": "not_applicable",
                },
                "telemetry": {
                    "records": _jsonable_records(records),
                    "advantages": advantage_summary,
                    "peak_vram_mib": _peak_vram_mib(torch, runtime.device),
                },
            }
            telemetry_file.write(json.dumps(payload, sort_keys=True) + "\n")
            telemetry_file.flush()

    del runtime.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lazy_torch() -> Any:
    import torch

    return torch


def run(config: RunnerConfig) -> None:
    """Run each requested arm from the same local base checkpoint."""
    if config.arm != "both":
        raise ValueError(
            "comparison pilot requires --arm both; single-arm output is not comparison-ready"
        )
    _enforce_offline()
    _enforce_determinism_environment()
    tasks = generate_canonical_splits(seed=config.seed)
    provenance = _provenance(config)
    manifest = _manifest_payload(config, tasks, provenance)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "status.json"
    terminal_path = output_dir / "terminal.jsonl"
    run_started = time.perf_counter()
    status: Dict[str, Any] = {
        "label": PILOT_LABEL,
        "run_status": "running",
        "comparison_complete": False,
        "failure_category": None,
        "pairing": {"checked": False, "status": "pending"},
        "arms": {
            arm: {
                "arm": arm,
                "status": "pending",
                "phase": None,
                "update": None,
                "duration_seconds": None,
                "peak_vram_mib": None,
                "failure_category": None,
                "exception_type": None,
                "exception_message": None,
            }
            for arm in config.arms
        },
    }
    manifest.update(
        {
            "run_status": "running",
            "comparison_complete": False,
            "terminal_status": None,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(status_path, status)

    pairing_state: Dict[str, Any] = {"checked": False, "status": "pending"}
    try:
        for arm in config.arms:
            arm_started = time.perf_counter()
            progress: Dict[str, Any] = {
                "phase": "load",
                "update": None,
                "peak_vram_mib": None,
            }
            status["arms"][arm] = {
                "arm": arm,
                "status": "running",
                "phase": "load",
                "update": None,
                "duration_seconds": None,
                "peak_vram_mib": None,
                "failure_category": None,
                "exception_type": None,
                "exception_message": None,
            }
            _write_json_atomic(status_path, status)
            try:
                _run_arm(
                    config,
                    arm,
                    tasks,
                    provenance,
                    pairing_state,
                    progress,
                )
            except BaseException as exc:
                terminal = _terminal_record(
                    arm=arm,
                    status="failed",
                    phase=progress["phase"],
                    update=progress["update"],
                    started=arm_started,
                    progress=progress,
                    exc=exc,
                )
                status["arms"][arm] = terminal
                status["run_status"] = "failed"
                status["failure_category"] = terminal["failure_category"]
                status["comparison_complete"] = False
                status["pairing"] = dict(pairing_state)
                status["duration_seconds"] = round(time.perf_counter() - run_started, 6)
                status["terminal_status"] = terminal
                _write_json_atomic(status_path, status)
                _append_jsonl(
                    terminal_path,
                    {
                        **provenance,
                        "label": PILOT_LABEL,
                        "comparison_complete": False,
                        "scope": "arm",
                        "terminal": terminal,
                    },
                )
                manifest.update(
                    {
                        "run_status": "failed",
                        "comparison_complete": False,
                        "failure_category": terminal["failure_category"],
                        "terminal_status": terminal,
                    }
                )
                _write_json_atomic(manifest_path, manifest)
                raise
            else:
                terminal = _terminal_record(
                    arm=arm,
                    status="completed",
                    phase="complete",
                    update=None,
                    started=arm_started,
                    progress=progress,
                )
                status["arms"][arm] = terminal
                status["pairing"] = dict(pairing_state)
                _write_json_atomic(status_path, status)
                _append_jsonl(
                    terminal_path,
                    {
                        **provenance,
                        "label": PILOT_LABEL,
                        "comparison_complete": False,
                        "scope": "arm",
                        "terminal": terminal,
                    },
                )
            finally:
                _write_json_atomic(status_path, status)

        if not pairing_state.get("checked", False):
            raise PairedRolloutMismatch(
                "paired comparison did not complete a pre-update fingerprint check"
            )
        status["run_status"] = "completed"
        status["comparison_complete"] = True
        status["duration_seconds"] = round(time.perf_counter() - run_started, 6)
        status["pairing"] = dict(pairing_state)
        run_terminal = _terminal_record(
            arm="run",
            status="completed",
            phase="complete",
            update=None,
            started=run_started,
            progress={"peak_vram_mib": None},
        )
        status["terminal_status"] = run_terminal
        _write_json_atomic(status_path, status)
        _append_jsonl(
            terminal_path,
            {
                **provenance,
                "label": PILOT_LABEL,
                "comparison_complete": True,
                "scope": "run",
                "terminal": run_terminal,
            },
        )
        manifest.update(
            {
                "run_status": "completed",
                "comparison_complete": True,
                "failure_category": None,
                "terminal_status": run_terminal,
            }
        )
        _write_json_atomic(manifest_path, manifest)
    except BaseException as exc:
        phase = "pairing" if isinstance(exc, PairedRolloutMismatch) else "run"
        run_terminal = _terminal_record(
            arm="run",
            status="failed",
            phase=phase,
            update=None,
            started=run_started,
            progress={"peak_vram_mib": None},
            exc=exc,
        )
        if status["run_status"] == "failed":
            run_terminal["failure_category"] = status["failure_category"]
        status["run_status"] = "failed"
        status["comparison_complete"] = False
        status["failure_category"] = run_terminal["failure_category"]
        status["pairing"] = dict(pairing_state)
        status["duration_seconds"] = round(time.perf_counter() - run_started, 6)
        status["terminal_status"] = run_terminal
        _write_json_atomic(status_path, status)
        _append_jsonl(
            terminal_path,
            {
                **provenance,
                "label": PILOT_LABEL,
                "comparison_complete": False,
                "scope": "run",
                "terminal": run_terminal,
            },
        )
        manifest.update(
            {
                "run_status": "failed",
                "comparison_complete": False,
                "failure_category": run_terminal["failure_category"],
                "terminal_status": run_terminal,
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise
    finally:
        status["duration_seconds"] = round(time.perf_counter() - run_started, 6)
        _write_json_atomic(status_path, status)


def _parse_args(argv: Optional[Sequence[str]] = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arm", choices=("both", *ARMS), default="both")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--opsd-granularity", choices=("turn", "token"), default="turn")
    args = parser.parse_args(argv)
    return RunnerConfig(**vars(args))


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = _parse_args(argv)
    run(config)
    print("[{}] completed output_dir={}".format(PILOT_LABEL, config.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
