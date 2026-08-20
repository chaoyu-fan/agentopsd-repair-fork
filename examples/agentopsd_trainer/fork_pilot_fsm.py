"""Asset-free CPU reference task for the labeled AgentOPSD engineering pilot.

This module is intentionally independent from Ray, FSDP, vLLM, and model
libraries. It defines only a deterministic task/environment contract plus the
adapter boundary where a later Qwen2.5-1.5B LoRA loop may provide generations.
It is not an upstream benchmark or a reproduction claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


PILOT_LABEL = "fork_engineering_pilot"
TASK_NAME = "fsm-code-v1"
MAX_TURNS = 3
MAX_NEW_TOKENS = 8
GROUP_SIZE = 4
ARMS = ("grpo", "agentopsd")
ACTION_CODES = ("A", "B", "C")
STATES = ("S0", "S1", "S2", "S3")
ACTION_PATTERN = re.compile(r"\AACTION ([ABC])\Z")


TASK_SCHEMA = {
    "$id": TASK_NAME,
    "type": "object",
    "required": ["action_order", "goal_state", "initial_state", "split", "task_id"],
    "properties": {
        "action_order": {
            "type": "array",
            "items": {"enum": list(ACTION_CODES)},
            "minItems": MAX_TURNS,
            "maxItems": MAX_TURNS,
        },
        "goal_state": {"const": "S3"},
        "initial_state": {"const": "S0"},
        "split": {"enum": ["train", "eval", "golden"]},
        "task_id": {"type": "string"},
    },
}


class ActionParseError(ValueError):
    """Raised when a model output does not match the exact action grammar."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FSMCodeTask:
    """A three-transition finite-state task with one advancing action per state."""

    task_id: str
    split: str
    action_order: Tuple[str, str, str]
    initial_state: str = "S0"
    goal_state: str = "S3"

    def __post_init__(self) -> None:
        if self.initial_state != STATES[0] or self.goal_state != STATES[-1]:
            raise ValueError("fsm-code-v1 requires S0 as initial state and S3 as goal state")
        if self.split not in ("train", "eval", "golden"):
            raise ValueError("split must be train, eval, or golden")
        if tuple(sorted(self.action_order)) != ACTION_CODES:
            raise ValueError("action_order must be a permutation of A, B, C")

    def canonical_dict(self) -> Dict[str, object]:
        return {
            "action_order": list(self.action_order),
            "goal_state": self.goal_state,
            "initial_state": self.initial_state,
            "split": self.split,
            "task_id": self.task_id,
        }

    @property
    def task_hash(self) -> str:
        return _sha256(self.canonical_dict())

    def next_state(self, state: str, action: str) -> str:
        """Advance only when action matches the code assigned to the state."""
        if state not in STATES:
            raise ValueError("unknown state: {!r}".format(state))
        if action not in ACTION_CODES:
            raise ValueError("unknown action: {!r}".format(action))
        if state == self.goal_state:
            return state
        state_index = STATES.index(state)
        if action == self.action_order[state_index]:
            return STATES[state_index + 1]
        return state

    def route_from(self, state: str) -> Tuple[str, ...]:
        if state not in STATES:
            raise ValueError("unknown state: {!r}".format(state))
        return self.action_order[STATES.index(state) :] if state != self.goal_state else ()


def generate_canonical_splits(seed: int = 0) -> Dict[str, Tuple[FSMCodeTask, ...]]:
    """Build fixed-size, deterministic train/eval splits without external assets."""
    action_orders = tuple(itertools.permutations(ACTION_CODES))
    split_specs = (("train", 8, 0), ("eval", 4, 3))
    splits: Dict[str, Tuple[FSMCodeTask, ...]] = {}
    for split, count, offset in split_specs:
        tasks = []
        for index in range(count):
            action_order = action_orders[(seed + offset + index) % len(action_orders)]
            task_id = "{}/{}/{:03d}".format(TASK_NAME, split, index)
            tasks.append(FSMCodeTask(task_id=task_id, split=split, action_order=action_order))
        splits[split] = tuple(tasks)
    return splits


def schema_hash() -> str:
    return _sha256(TASK_SCHEMA)


def split_hash(tasks: Sequence[FSMCodeTask]) -> str:
    """Hash a split's canonical rows in their generated order."""
    return _sha256([task.canonical_dict() for task in tasks])


def split_hashes(seed: int = 0) -> Dict[str, str]:
    return {name: split_hash(tasks) for name, tasks in generate_canonical_splits(seed).items()}


def parse_action(text: str) -> str:
    """Accept exactly ``ACTION <A|B|C>`` with no leading/trailing characters."""
    if not isinstance(text, str):
        raise ActionParseError("action output must be str")
    match = ACTION_PATTERN.fullmatch(text)
    if match is None:
        raise ActionParseError("expected exact `ACTION <A|B|C>`, got {!r}".format(text))
    return match.group(1)


def student_view(task: FSMCodeTask, state: str, turn: int) -> str:
    """Render the public task view used by the rollout adapter."""
    if state not in STATES:
        raise ValueError("unknown state: {!r}".format(state))
    if not 0 <= turn < MAX_TURNS:
        raise ValueError("turn must be in [0, {})".format(MAX_TURNS))

    transition_lines = []
    for index, source in enumerate(STATES[:-1]):
        advance_action = task.action_order[index]
        transition_lines.append(
            "{} + {} -> {}; all other actions remain at {}.".format(
                source, advance_action, STATES[index + 1], source
            )
        )
    return "\n".join(
        (
            "Task: {}".format(TASK_NAME),
            "Current state: {}".format(state),
            "Goal state: {}".format(task.goal_state),
            "Turn: {}/{}".format(turn + 1, MAX_TURNS),
            "Transitions:",
            *transition_lines,
            "Reply exactly: ACTION <A|B|C>",
        )
    )


def oracle_skill_view(task: FSMCodeTask, state: str) -> str:
    """Render the privileged route used only by a future teacher-side adapter."""
    route = task.route_from(state)
    actions = " -> ".join("ACTION {}".format(action) for action in route) or "already complete"
    return "\n".join(
        (
            "Oracle skill for {}.".format(TASK_NAME),
            "State: {}; goal: {}.".format(state, task.goal_state),
            "Remaining exact route: {}.".format(actions),
        )
    )


@dataclass(frozen=True)
class StepTelemetry:
    turn: int
    state_before: str
    action_text: str
    parsed_action: Optional[str]
    state_after: str
    reward: float
    terminated: bool
    parse_error: Optional[str] = None


@dataclass
class FSMCodeEnvironment:
    """A stateful, three-turn environment with a sparse terminal reward."""

    task: FSMCodeTask
    state: str = field(init=False)
    turn: int = field(default=0, init=False)
    terminated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.state = self.task.initial_state

    def step(self, action_text: str) -> StepTelemetry:
        if self.terminated:
            raise RuntimeError("cannot step a terminated fsm-code-v1 environment")

        state_before = self.state
        parse_error = None
        parsed_action = None
        try:
            parsed_action = parse_action(action_text)
            self.state = self.task.next_state(self.state, parsed_action)
        except ActionParseError as exc:
            parse_error = str(exc)
            self.turn += 1
            self.terminated = True
            return StepTelemetry(
                turn=self.turn,
                state_before=state_before,
                action_text=action_text,
                parsed_action=None,
                state_after=self.state,
                reward=0.0,
                terminated=True,
                parse_error=parse_error,
            )

        self.turn += 1
        reward = 1.0 if self.state == self.task.goal_state else 0.0
        self.terminated = reward == 1.0 or self.turn >= MAX_TURNS
        return StepTelemetry(
            turn=self.turn,
            state_before=state_before,
            action_text=action_text,
            parsed_action=parsed_action,
            state_after=self.state,
            reward=reward,
            terminated=self.terminated,
            parse_error=parse_error,
        )


@dataclass(frozen=True)
class PilotConfig:
    """Fixed limits for the explicitly labeled engineering pilot."""

    arm: str
    label: str = PILOT_LABEL
    max_turns: int = MAX_TURNS
    max_new_tokens: int = MAX_NEW_TOKENS
    group_size: int = GROUP_SIZE

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("arm must be one of {}".format(ARMS))
        if self.label != PILOT_LABEL:
            raise ValueError("pilot label must be {!r}".format(PILOT_LABEL))
        if self.max_turns != MAX_TURNS:
            raise ValueError("max_turns is fixed at {}".format(MAX_TURNS))
        if self.max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError("max_new_tokens is fixed at {}".format(MAX_NEW_TOKENS))
        if self.group_size != GROUP_SIZE:
            raise ValueError("group_size is fixed at {}".format(GROUP_SIZE))


@dataclass(frozen=True)
class RolloutTelemetry:
    task_id: str
    arm: str
    rollout_index: int
    steps: Tuple[StepTelemetry, ...]
    final_state: str
    reward: float

    @property
    def success(self) -> bool:
        return self.reward == 1.0


@dataclass(frozen=True)
class GroupTelemetry:
    label: str
    arm: str
    task_id: str
    task_hash: str
    schema_sha256: str
    rollouts: Tuple[RolloutTelemetry, ...]

    @property
    def rewards(self) -> Tuple[float, ...]:
        return tuple(rollout.reward for rollout in self.rollouts)


class ActionGenerator(Protocol):
    """Coordinator integration point for a future Qwen2.5-1.5B LoRA adapter."""

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> Sequence[str]:
        """Return exactly one action string for each supplied student prompt."""


class PilotHarness:
    """CPU-only rollout orchestrator; it does not import or invoke a trainer."""

    def __init__(self, config: PilotConfig):
        self.config = config

    def run_group(self, task: FSMCodeTask, generator: ActionGenerator) -> GroupTelemetry:
        environments = [FSMCodeEnvironment(task) for _ in range(self.config.group_size)]
        traces: List[List[StepTelemetry]] = [[] for _ in environments]

        while True:
            active_indices = [index for index, environment in enumerate(environments) if not environment.terminated]
            if not active_indices:
                break
            prompts = [
                student_view(environments[index].task, environments[index].state, environments[index].turn)
                for index in active_indices
            ]
            outputs = list(generator.generate(prompts=prompts, max_new_tokens=self.config.max_new_tokens))
            if len(outputs) != len(active_indices):
                raise ValueError(
                    "generator returned {} outputs for {} active environments".format(
                        len(outputs), len(active_indices)
                    )
                )
            for index, output in zip(active_indices, outputs):
                traces[index].append(environments[index].step(output))

        rollouts = tuple(
            RolloutTelemetry(
                task_id=task.task_id,
                arm=self.config.arm,
                rollout_index=index,
                steps=tuple(traces[index]),
                final_state=environment.state,
                reward=1.0 if environment.state == task.goal_state else 0.0,
            )
            for index, environment in enumerate(environments)
        )
        return GroupTelemetry(
            label=self.config.label,
            arm=self.config.arm,
            task_id=task.task_id,
            task_hash=task.task_hash,
            schema_sha256=schema_hash(),
            rollouts=rollouts,
        )


class _GoldenGenerator:
    """Deterministic generator used only by the asset-free CPU goldens."""

    def __init__(self, actions_by_turn: Sequence[Sequence[str]]):
        self.actions_by_turn = tuple(tuple(actions) for actions in actions_by_turn)
        self.turn = 0

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> Sequence[str]:
        if max_new_tokens != MAX_NEW_TOKENS:
            raise AssertionError("golden generator received unexpected token limit")
        outputs = self.actions_by_turn[self.turn]
        self.turn += 1
        if len(outputs) < len(prompts):
            raise AssertionError("golden generator output count does not match prompts")
        return outputs[: len(prompts)]


CPU_GOLDENS = {
    "schema_sha256": "f7169e3c352869c18b20d3d14e670c453126d29b784f4ed5f2b99cae91539fe5",
    "train_sha256": "924bc6feebe3b58d265f5f70ad5c1874869338116c9a596de9643cc37f3ad78d",
    "eval_sha256": "00ecf45e2b4301a5a6670b312512415a1cfc3204031203c291047f99c0e24d58",
    "trace_sha256": "c52c58bbf3c79ad33e052e25c047bd8b3c0a9b88b91557b738a7d3e5a034bf5f",
}


def _trace_value(telemetry: GroupTelemetry) -> Dict[str, object]:
    return {
        "arm": telemetry.arm,
        "label": telemetry.label,
        "rewards": list(telemetry.rewards),
        "rollouts": [
            {
                "final_state": rollout.final_state,
                "steps": [
                    {
                        "action_text": step.action_text,
                        "parse_error": step.parse_error,
                        "parsed_action": step.parsed_action,
                        "state_after": step.state_after,
                        "state_before": step.state_before,
                        "turn": step.turn,
                    }
                    for step in rollout.steps
                ],
            }
            for rollout in telemetry.rollouts
        ],
        "task_hash": telemetry.task_hash,
        "task_id": telemetry.task_id,
    }


def run_cpu_goldens() -> Dict[str, str]:
    """Check fixed parser, schema/split hashes, and a complete 3-turn group trace."""
    splits = generate_canonical_splits()
    golden_task = FSMCodeTask(
        task_id="{}/golden/000".format(TASK_NAME),
        split="golden",
        action_order=("A", "B", "C"),
    )
    generator = _GoldenGenerator(
        (
            ("ACTION A", "ACTION B", "ACTION A", "ACTION A "),
            ("ACTION B", "ACTION B", "ACTION B", "ACTION B"),
            ("ACTION C", "ACTION C", "ACTION X", "ACTION C"),
        )
    )
    telemetry = PilotHarness(PilotConfig(arm="agentopsd")).run_group(golden_task, generator)
    values = {
        "schema_sha256": schema_hash(),
        "train_sha256": split_hash(splits["train"]),
        "eval_sha256": split_hash(splits["eval"]),
        "trace_sha256": _sha256(_trace_value(telemetry)),
    }
    if telemetry.rewards != (1.0, 0.0, 0.0, 0.0):
        raise AssertionError("unexpected golden rewards: {!r}".format(telemetry.rewards))
    if telemetry.rollouts[2].steps[-1].parse_error is None:
        raise AssertionError("invalid action golden did not record a parser error")
    if telemetry.rollouts[3].steps[0].parse_error is None:
        raise AssertionError("trailing-space golden did not record a parser error")
    if parse_action("ACTION B") != "B":
        raise AssertionError("exact parser golden failed")
    for invalid in ("ACTION B ", "action B", "ACTION D", "ACTION B\n"):
        try:
            parse_action(invalid)
        except ActionParseError:
            pass
        else:
            raise AssertionError("parser accepted invalid action {!r}".format(invalid))
    if oracle_skill_view(golden_task, "S0").splitlines()[-1] != "Remaining exact route: ACTION A -> ACTION B -> ACTION C.":
        raise AssertionError("oracle route golden failed")
    if split_hashes(seed=1) == split_hashes(seed=0):
        raise AssertionError("split hash did not change when canonical task seed changed")
    for name, expected in CPU_GOLDENS.items():
        if values[name] != expected:
            raise AssertionError("{} changed: {} != {}".format(name, values[name], expected))
    return values


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic CPU goldens")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported; model/training execution is intentionally absent")
    values = run_cpu_goldens()
    print("fsm-code-v1 CPU goldens: PASS")
    for name in sorted(values):
        print("{}={}".format(name, values[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
