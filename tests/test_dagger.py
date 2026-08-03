from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from hierarchical_minivla.dagger import (
    DESCEND_SETTLE_STEPS,
    REACH_SETTLE_STEPS,
    ScriptedRecoveryOracle,
    run_dagger_episode,
)
from hierarchical_minivla.state_policy import GRIPPER_OPEN


class FakeEnv:
    def __init__(self) -> None:
        self.mocap_id = 0
        self.data = SimpleNamespace(mocap_pos=np.array([[-0.23, 0.50, 0.12]]))

    def get_target_cube_state(self) -> np.ndarray:
        return np.array([-0.20, 0.55, 0.02, 1.0, 0.0, 0.0, 0.0])

    def get_goal_pos(self) -> np.ndarray:
        return np.array([-0.20, 0.70, 0.021])

    def get_gripper_angle(self) -> float:
        return 1.0


class DaggerTest(unittest.TestCase):
    def test_reach_label_is_bounded_and_keeps_gripper_open(self) -> None:
        delta, gripper, phase = ScriptedRecoveryOracle().label(FakeEnv())
        self.assertLessEqual(float(np.abs(delta).max()), 0.004)
        self.assertEqual(gripper, GRIPPER_OPEN)
        self.assertEqual(phase, 0)

    def test_recovery_oracle_settles_before_reach_and_descend_transitions(self) -> None:
        env = FakeEnv()
        oracle = ScriptedRecoveryOracle()
        env.data.mocap_pos[0] = oracle.target(env)

        for _ in range(REACH_SETTLE_STEPS):
            self.assertFalse(oracle.maybe_advance(env))
        self.assertTrue(oracle.maybe_advance(env))
        self.assertEqual(oracle.phase, 1)

        env.data.mocap_pos[0] = oracle.target(env)
        for _ in range(DESCEND_SETTLE_STEPS):
            self.assertFalse(oracle.maybe_advance(env))
        self.assertTrue(oracle.maybe_advance(env))
        self.assertEqual(oracle.phase, 2)

    def test_recovery_oracle_resets_settle_count_after_leaving_target(self) -> None:
        env = FakeEnv()
        oracle = ScriptedRecoveryOracle()
        target = oracle.target(env)
        env.data.mocap_pos[0] = target
        self.assertFalse(oracle.maybe_advance(env))

        env.data.mocap_pos[0] = target + np.array([0.01, 0.0, 0.0])
        self.assertFalse(oracle.maybe_advance(env))
        self.assertEqual(oracle.target_settle_steps, 0)

    def test_negative_grasp_lift_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "learner_grasp_lift_steps"):
            run_dagger_episode(None, None, learner_grasp_lift_steps=-1)

    def test_negative_transport_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "learner_transport_steps"):
            run_dagger_episode(None, None, learner_transport_steps=-1)


if __name__ == "__main__":
    unittest.main()
