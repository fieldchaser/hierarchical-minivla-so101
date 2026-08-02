from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hierarchical_minivla.hierarchical_policy import (
    HierarchicalStateGoalPolicy,
    MonotonicPhaseTracker,
    balanced_phase_weights,
    load_hierarchical_policy,
    observed_event_next_phase,
    phase_progress_summary,
    save_hierarchical_policy,
)


def make_policy() -> HierarchicalStateGoalPolicy:
    return HierarchicalStateGoalPolicy(
        np.zeros(34),
        np.ones(34),
        np.zeros(3),
        np.ones(3),
        np.ones(6),
        hidden_dim=16,
    )


class HierarchicalPolicyTest(unittest.TestCase):
    def test_phase_weights_upweight_rare_classes(self) -> None:
        weights = balanced_phase_weights(np.array([0, 0, 0, 1, 2, 3, 4, 5]))
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_phase_tracker_requires_votes_and_never_moves_backward(self) -> None:
        tracker = MonotonicPhaseTracker(required_votes=2)
        self.assertEqual(tracker.update(1), 0)
        self.assertEqual(tracker.update(1), 1)
        self.assertEqual(tracker.update(0), 1)
        self.assertEqual(tracker.update(5), 1)
        self.assertEqual(tracker.update(5), 1)
        self.assertEqual(tracker.update(2), 1)
        self.assertEqual(tracker.update(2), 2)

    def test_phase_progress_summary_reports_terminal_and_reached_counts(self) -> None:
        summary = phase_progress_summary([0, 1, 1, 3])
        self.assertEqual(summary["mean_final_phase"], 1.25)
        self.assertEqual(summary["final_phase_counts"]["descend"], 2)
        self.assertEqual(summary["reached_phase_counts"]["descend"], 3)
        self.assertEqual(summary["reached_phase_counts"]["grasp"], 1)

    def test_observed_events_advance_reach_and_descend(self) -> None:
        observation = {
            "cubes_xyz": np.array(
                [-0.2, 0.35, 0.02, -0.2, 0.45, 0.02, -0.2, 0.55, 0.02]
            ),
            "goal": np.array([0.0, 1.0, 0.0]),
            "goal_pos": np.array([-0.2, 0.7, 0.021]),
            "gripper": np.array([1.0]),
        }
        above_cube = np.array([-0.215, 0.446, 0.15])
        grasp_target = np.array([-0.215, 0.446, 0.078])
        self.assertEqual(observed_event_next_phase(observation, above_cube, 0, 4), 1)
        self.assertEqual(
            observed_event_next_phase(observation, grasp_target, 1, 4), 2
        )

    def test_observed_events_advance_contact_lift_and_transport(self) -> None:
        observation = {
            "cubes_xyz": np.array(
                [-0.2, 0.35, 0.09, -0.2, 0.45, 0.02, -0.2, 0.55, 0.02]
            ),
            "goal": np.array([1.0, 0.0, 0.0]),
            "goal_pos": np.array([-0.2, 0.35, 0.021]),
            "gripper": np.array([0.5]),
        }
        mocap = np.array([-0.215, 0.346, 0.17])
        self.assertEqual(observed_event_next_phase(observation, mocap, 2, 20), 3)
        self.assertEqual(observed_event_next_phase(observation, mocap, 3, 1), 4)
        self.assertEqual(observed_event_next_phase(observation, mocap, 4, 1), 5)

    def test_checkpoint_round_trip_preserves_phase_and_action(self) -> None:
        policy = make_policy()
        feature = np.linspace(-1.0, 1.0, 34, dtype=np.float32)
        expected = policy.act(feature, phase=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_hierarchical_policy(policy, Path(tmp) / "policy.pt")
            loaded = load_hierarchical_policy(path)
        actual = loaded.act(feature, phase=2)
        np.testing.assert_allclose(actual[0], expected[0])
        self.assertEqual(actual[1:], expected[1:])


if __name__ == "__main__":
    unittest.main()
