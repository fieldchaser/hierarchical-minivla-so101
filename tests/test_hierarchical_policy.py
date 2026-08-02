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
