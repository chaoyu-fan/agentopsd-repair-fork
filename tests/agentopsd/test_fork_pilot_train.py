"""CPU-only checks for standalone fork pilot runner metadata."""

import os
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from examples.agentopsd_trainer import fork_pilot_fsm as fsm
from examples.agentopsd_trainer import fork_pilot_train as runner


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "examples/agentopsd_trainer/fork_pilot_train.py"


class TestForkPilotTrain(unittest.TestCase):
    def _write_local_model_metadata(self, root: Path) -> None:
        (root / "model.safetensors").write_bytes(b"local model weights")
        (root / "tokenizer.json").write_text('{"model":"test"}\n', encoding="utf-8")
        (root / "tokenizer_config.json").write_text('{"pad_token":"<pad>"}\n', encoding="utf-8")
        (root / "config.json").write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
        (root / "generation_config.json").write_text('{"max_new_tokens":8}\n', encoding="utf-8")

    def _config(self, model_path: Path, arm: str = "both") -> runner.RunnerConfig:
        return runner.RunnerConfig(
            model_path=str(model_path),
            arm=arm,
            device="cpu",
            dtype="fp32",
            seed=17,
            updates=2,
        )

    def test_direct_script_help_uses_import_fallback_without_model_load(self):
        result = subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--help"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Standalone paired LoRA pilot runner", result.stdout)
        self.assertIn("--model-path", result.stdout)

    def test_provenance_enforces_offline_and_hashes_local_assets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            self._write_local_model_metadata(model_path)
            config = self._config(model_path)

            with patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
                clear=False,
            ):
                provenance = runner._provenance(config)
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

            self.assertEqual(provenance["offline"], {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "local_files_only": True,
            })
            self.assertEqual(provenance["split_hashes"], fsm.split_hashes(seed=17))
            self.assertEqual(provenance["schema_sha256"], fsm.schema_hash())
            self.assertEqual(provenance["paired_arm_invariants"]["seed"], 17)
            self.assertEqual(provenance["paired_arm_invariants"]["model_path"], str(model_path))
            self.assertTrue(provenance["git_commit"])
            for name in (
                "model_aggregate_sha256",
                "tokenizer_metadata_sha256",
                "config_metadata_sha256",
                "source_aggregate_sha256",
            ):
                self.assertRegex(provenance[name], r"^[0-9a-f]{64}$")
            self.assertEqual(
                sorted(provenance["source_hashes"]),
                ["fsm_sha256", "opsd_utils_sha256", "runner_sha256"],
            )

    def test_manifest_and_record_have_shared_paired_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            self._write_local_model_metadata(model_path)
            config = self._config(model_path)
            tasks = fsm.generate_canonical_splits(seed=config.seed)
            provenance = runner._provenance(config)
            manifest = runner._manifest_payload(config, tasks, provenance)
            train_task = tasks["train"][0]
            record = runner._record_provenance(
                config, train_task, "train", 0, provenance
            )

            self.assertIsNone(manifest["failure_category"])
            self.assertIsNone(record["failure_category"])
            self.assertEqual(manifest["git_commit"], record["git_commit"])
            self.assertEqual(manifest["split_hashes"], record["split_hashes"])
            self.assertEqual(manifest["offline"], record["offline"])
            self.assertRegex(manifest["rollout_plan_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(record["rollout_sha256"], r"^[0-9a-f]{64}$")

            grpo_config = self._config(model_path, arm="grpo")
            opsd_config = self._config(model_path, arm="agentopsd")
            grpo_fields = asdict(grpo_config)
            opsd_fields = asdict(opsd_config)
            self.assertEqual(grpo_fields.pop("arm"), "grpo")
            self.assertEqual(opsd_fields.pop("arm"), "agentopsd")
            self.assertEqual(grpo_fields, opsd_fields)
            self.assertEqual(
                runner._rollout_sha256(grpo_config, train_task, "train", 0),
                runner._rollout_sha256(opsd_config, train_task, "train", 0),
            )
            self.assertEqual(manifest["v0_strategy"], "group_mean")
            self.assertNotIn("opsd_v0_prior", manifest)
            self.assertEqual(
                manifest["determinism_policy"]["torch_use_deterministic_algorithms"],
                True,
            )

    def test_run_rejects_single_arm_comparison_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = self._config(Path(temporary_directory), arm="grpo")
            with self.assertRaisesRegex(ValueError, "requires --arm both"):
                runner.run(config)

    def test_cli_missing_local_model_persists_preflight_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "output"
            missing_model_path = root / "missing-model"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--arm",
                    "both",
                    "--model-path",
                    str(missing_model_path),
                    "--output-dir",
                    str(output_path),
                    "--device",
                    "cpu",
                    "--dtype",
                    "fp32",
                ],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            manifest_path = output_path / "manifest.json"
            status_path = output_path / "status.json"
            terminal_path = output_path / "terminal.jsonl"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(status_path.is_file())
            self.assertTrue(terminal_path.is_file())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            terminal_events = [
                json.loads(line)
                for line in terminal_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest["run_status"], "failed")
            self.assertFalse(manifest["comparison_complete"])
            self.assertEqual(manifest["failure_category"], "model_load")
            self.assertEqual(status["run_status"], "failed")
            self.assertFalse(status["comparison_complete"])
            self.assertEqual(status["failure_category"], "model_load")
            self.assertEqual(len(terminal_events), 1)
            terminal = terminal_events[0]["terminal"]
            self.assertEqual(terminal_events[0]["scope"], "run")
            self.assertEqual(terminal["phase"], "preflight")
            self.assertEqual(terminal["failure_category"], "model_load")
            self.assertEqual(terminal["exception_type"], "FileNotFoundError")
            self.assertIn(str(missing_model_path), terminal["exception_message"])

    def test_existing_pilot_artifact_fails_closed_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "output"
            output_path.mkdir()
            manifest_path = output_path / "manifest.json"
            manifest_path.write_text("existing pilot manifest\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--arm",
                    "both",
                    "--model-path",
                    "/definitely/not/a/local/model",
                    "--output-dir",
                    str(output_path),
                    "--device",
                    "cpu",
                    "--dtype",
                    "fp32",
                ],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already contains pilot artifacts", result.stderr)
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"), "existing pilot manifest\n"
            )
            self.assertFalse((output_path / "status.json").exists())
            self.assertFalse((output_path / "terminal.jsonl").exists())

    def test_failed_arm_persists_terminal_status_and_blocks_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model"
            model_path.mkdir()
            self._write_local_model_metadata(model_path)
            output_path = Path(temporary_directory) / "output"
            config = runner.RunnerConfig(
                model_path=str(model_path),
                output_dir=str(output_path),
                arm="both",
                device="cpu",
                dtype="fp32",
            )
            with patch.object(
                runner, "_load_runtime", side_effect=RuntimeError("synthetic load failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
                    runner.run(config)

            status = json.loads((output_path / "status.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (output_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["run_status"], "failed")
            self.assertFalse(status["comparison_complete"])
            self.assertEqual(status["arms"]["grpo"]["status"], "failed")
            self.assertEqual(status["arms"]["grpo"]["exception_type"], "RuntimeError")
            self.assertEqual(status["arms"]["grpo"]["exception_message"], "synthetic load failure")
            self.assertEqual(manifest["run_status"], "failed")
            self.assertFalse(manifest["comparison_complete"])
            self.assertEqual(manifest["failure_category"], "model_load")
            terminal_events = [
                json.loads(line)
                for line in (output_path / "terminal.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([event["scope"] for event in terminal_events], ["arm", "run"])
            self.assertEqual(terminal_events[0]["terminal"]["arm"], "grpo")
            self.assertEqual(terminal_events[0]["terminal"]["status"], "failed")
            self.assertEqual(terminal_events[1]["terminal"]["arm"], "run")
            self.assertEqual(terminal_events[1]["terminal"]["failure_category"], "model_load")

    def test_sampling_seed_and_fingerprint_are_arm_independent(self):
        task = fsm.generate_canonical_splits(seed=0)["train"][0]
        grpo = self._config(Path("/tmp/grpo"), arm="grpo")
        agentopsd = self._config(Path("/tmp/agentopsd"), arm="agentopsd")
        grpo_seed = runner._sampling_seed(grpo, task, "train", 0, 2, 1)
        opsd_seed = runner._sampling_seed(agentopsd, task, "train", 0, 2, 1)
        self.assertEqual(grpo_seed, opsd_seed)
        self.assertNotEqual(grpo_seed, runner._sampling_seed(grpo, task, "train", 0, 2, 2))
        first = runner._completion_fingerprint([1, 2, 3])
        second = runner._completion_fingerprint([1, 2, 4])
        self.assertNotEqual(first, second)
        runner._compare_paired_fingerprints({0: first}, {0: first})
        with self.assertRaises(runner.PairedRolloutMismatch):
            runner._compare_paired_fingerprints({0: first}, {0: second})

    def test_paired_completion_requires_a_matched_preupdate_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model"
            model_path.mkdir()
            self._write_local_model_metadata(model_path)
            output_path = Path(temporary_directory) / "output"
            config = runner.RunnerConfig(
                model_path=str(model_path),
                output_dir=str(output_path),
                arm="both",
                device="cpu",
                dtype="fp32",
            )

            def fake_run_arm(config, arm, tasks, provenance, pairing_state, progress):
                progress.update({"phase": "eval", "update": 3, "peak_vram_mib": None})
                fingerprints = {0: "same", 1: "same", 2: "same", 3: "same"}
                if arm == "grpo":
                    pairing_state.update(
                        {"reference": fingerprints, "status": "reference_recorded"}
                    )
                else:
                    runner._compare_paired_fingerprints(
                        pairing_state["reference"], fingerprints
                    )
                    pairing_state.update(
                        {
                            "candidate": fingerprints,
                            "checked": True,
                            "status": "matched",
                        }
                    )

            with patch.object(runner, "_run_arm", side_effect=fake_run_arm):
                runner.run(config)

            status = json.loads((output_path / "status.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (output_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["run_status"], "completed")
            self.assertTrue(status["comparison_complete"])
            self.assertEqual(status["pairing"]["status"], "matched")
            self.assertEqual(manifest["run_status"], "completed")
            self.assertTrue(manifest["comparison_complete"])
            self.assertEqual(
                [event["scope"] for event in map(
                    json.loads,
                    (output_path / "terminal.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines(),
                )],
                ["arm", "arm", "run"],
            )

    def test_determinism_policy_is_enforced_before_runtime_load(self):
        class FakeCudnn:
            deterministic = None
            benchmark = None

        class FakeMatmul:
            allow_tf32 = None

        class FakeCuda:
            matmul = FakeMatmul()

        class FakeBackends:
            cudnn = FakeCudnn()
            cuda = FakeCuda()

        class FakeTorch:
            backends = FakeBackends()

            def __init__(self):
                self.calls = []

            def use_deterministic_algorithms(self, enabled):
                self.calls.append(enabled)

        fake_torch = FakeTorch()
        policy = runner._configure_determinism(fake_torch)
        self.assertEqual(fake_torch.calls, [True])
        self.assertTrue(fake_torch.backends.cudnn.deterministic)
        self.assertFalse(fake_torch.backends.cudnn.benchmark)
        self.assertFalse(fake_torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(
            os.environ["CUBLAS_WORKSPACE_CONFIG"],
            policy["cublas_workspace_config"],
        )


if __name__ == "__main__":
    unittest.main()
