from __future__ import annotations

import unittest

import numpy as np

from hierarchical_minivla.scripted_expert import (
    PHASE_NAMES,
    bounded_delta,
    is_released_in_bin,
    run_scripted_episode,
)


class ScriptedExpertTest(unittest.TestCase):
    def test_phase_order_matches_manipulation_sequence(self) -> None:
        self.assertEqual(
            PHASE_NAMES,
            ("reach", "descend", "grasp", "lift", "transport", "release"),
        )

    def test_cartesian_delta_is_bounded(self) -> None:
        delta = bounded_delta(np.zeros(3), np.array([1.0, -1.0, 0.002]))
        np.testing.assert_allclose(delta, [0.004, -0.004, 0.002])

    def test_success_requires_cube_in_bin_and_open_gripper(self) -> None:
        cube = np.array([-0.2, 0.7, 0.02])
        bin_position = np.array([-0.2, 0.7, 0.021])
        self.assertTrue(is_released_in_bin(cube, bin_position, gripper_angle=1.0))
        self.assertFalse(is_released_in_bin(cube, bin_position, gripper_angle=0.3))
        self.assertFalse(
            is_released_in_bin(cube + np.array([0.1, 0.0, 0.0]), bin_position, 1.0)
        )

    def test_recovery_noise_requires_explicit_rng(self) -> None:
        with self.assertRaisesRegex(ValueError, "recovery_rng"):
            run_scripted_episode(object(), recovery_pos_std=0.005)


if __name__ == "__main__":
    unittest.main()
