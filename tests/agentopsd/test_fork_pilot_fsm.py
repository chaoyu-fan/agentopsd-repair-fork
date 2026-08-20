"""Focused CPU goldens for the asset-free fsm-code-v1 pilot."""

import unittest

from examples.agentopsd_trainer import fork_pilot_fsm as fsm


class TestForkPilotFSM(unittest.TestCase):
    def test_cpu_goldens_are_frozen(self):
        self.assertEqual(fsm.run_cpu_goldens(), fsm.CPU_GOLDENS)

    def test_parser_is_exact(self):
        self.assertEqual(fsm.parse_action("ACTION C"), "C")
        for invalid in ("ACTION C ", " ACTION C", "ACTION D", "action C", "ACTION C\n"):
            with self.assertRaises(fsm.ActionParseError):
                fsm.parse_action(invalid)

    def test_three_turn_environment_and_oracle_route(self):
        task = fsm.FSMCodeTask(
            task_id="fsm-code-v1/golden/test",
            split="golden",
            action_order=("A", "B", "C"),
        )
        env = fsm.FSMCodeEnvironment(task)
        self.assertEqual(task.route_from("S0"), ("A", "B", "C"))
        self.assertIn("ACTION A -> ACTION B -> ACTION C", fsm.oracle_skill_view(task, "S0"))

        self.assertEqual(env.step("ACTION A").state_after, "S1")
        self.assertEqual(env.step("ACTION B").state_after, "S2")
        final_step = env.step("ACTION C")
        self.assertEqual(final_step.state_after, "S3")
        self.assertTrue(final_step.terminated)
        self.assertEqual(final_step.reward, 1.0)
        with self.assertRaises(RuntimeError):
            env.step("ACTION A")

    def test_invalid_action_terminates_immediately_with_zero_reward(self):
        task = fsm.FSMCodeTask(
            task_id="fsm-code-v1/golden/invalid",
            split="golden",
            action_order=("A", "B", "C"),
        )
        env = fsm.FSMCodeEnvironment(task)
        step = env.step("ACTION A ")
        self.assertEqual(step.turn, 1)
        self.assertEqual(step.state_before, "S0")
        self.assertEqual(step.state_after, "S0")
        self.assertEqual(step.reward, 0.0)
        self.assertTrue(step.terminated)
        self.assertIsNotNone(step.parse_error)
        with self.assertRaises(RuntimeError):
            env.step("ACTION A")

    def test_schema_is_fixed_while_split_hash_tracks_seed(self):
        self.assertEqual(fsm.schema_hash(), fsm.CPU_GOLDENS["schema_sha256"])
        self.assertEqual(fsm.split_hashes(0)["train"], fsm.CPU_GOLDENS["train_sha256"])
        self.assertNotEqual(fsm.split_hashes(0), fsm.split_hashes(1))

    def test_pilot_limits_and_arms_are_fixed(self):
        for arm in fsm.ARMS:
            config = fsm.PilotConfig(arm=arm)
            self.assertEqual(config.label, "fork_engineering_pilot")
            self.assertEqual(config.max_turns, 3)
            self.assertEqual(config.max_new_tokens, 8)
            self.assertEqual(config.group_size, 4)
        with self.assertRaises(ValueError):
            fsm.PilotConfig(arm="ppo")
        with self.assertRaises(ValueError):
            fsm.PilotConfig(arm="grpo", max_turns=4)


if __name__ == "__main__":
    unittest.main()
