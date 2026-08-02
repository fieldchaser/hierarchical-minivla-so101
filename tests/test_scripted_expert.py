from __future__ import annotations

import unittest

import numpy as np

from hierarchical_minivla.scripted_expert import (
    INSTRUCTION_TEMPLATES,
    PHASE_NAMES,
    bounded_delta,
    instruction_for_goal,
    instruction_variant_for_episode,
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

    def test_instruction_variants_preserve_goal_color(self) -> None:
        instructions = {
            instruction_for_goal("green", variant)
            for variant in range(len(INSTRUCTION_TEMPLATES))
        }
        self.assertEqual(len(instructions), len(INSTRUCTION_TEMPLATES))
        self.assertTrue(all("green" in text for text in instructions))

    def test_instruction_schedule_covers_every_template_for_every_color(self) -> None:
        colors = ("red", "green", "blue")
        instructions_by_color = {color: set() for color in colors}
        for episode_index in range(len(colors) * len(INSTRUCTION_TEMPLATES)):
            color = colors[episode_index % len(colors)]
            variant = instruction_variant_for_episode(episode_index, len(colors))
            instructions_by_color[color].add(instruction_for_goal(color, variant))
        self.assertTrue(
            all(
                len(instructions) == len(INSTRUCTION_TEMPLATES)
                for instructions in instructions_by_color.values()
            )
        )

    def test_rgb_recording_requires_instruction(self) -> None:
        with self.assertRaisesRegex(ValueError, "instruction"):
            run_scripted_episode(
                object(), frame_observer=lambda: np.zeros((8, 8, 3), np.uint8)
            )


if __name__ == "__main__":
    unittest.main()
