from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hierarchical_minivla.state_policy import (
    StateGoalPolicy,
    load_policy,
    observation_to_feature,
    save_policy,
    split_episode_paths,
)


class StatePolicyTest(unittest.TestCase):
    def test_episode_split_has_no_overlap(self) -> None:
        paths = [Path(f"episode_{index}.npz") for index in range(10)]
        train, validation = split_episode_paths(paths, seed=7)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse(set(train) & set(validation))

    def test_observation_feature_has_expected_shape(self) -> None:
        observation = {
            "ee_pos": np.zeros(3),
            "joints": np.zeros(6),
            "gripper": np.zeros(1),
            "cubes_xyz": np.zeros(9),
            "goal": np.zeros(3),
            "goal_pos": np.zeros(3),
        }
        self.assertEqual(observation_to_feature(observation).shape, (34,))
        self.assertEqual(
            observation_to_feature(observation, np.zeros(3)).shape,
            (37,),
        )

    def test_checkpoint_round_trip_preserves_action(self) -> None:
        policy = StateGoalPolicy(
            np.zeros(34), np.ones(34), np.zeros(3), np.ones(3), hidden_dim=16
        )
        feature = np.linspace(-1.0, 1.0, 34, dtype=np.float32)
        expected = policy.act(feature)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_policy(policy, Path(tmp) / "policy.pt")
            loaded = load_policy(path)
        actual = loaded.act(feature)
        np.testing.assert_allclose(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])

    def test_action_delta_is_clipped(self) -> None:
        policy = StateGoalPolicy(
            np.zeros(34), np.ones(34), np.zeros(3), np.ones(3), hidden_dim=8
        )
        with torch.no_grad():
            policy.network[-1].bias[:3].fill_(100.0)
        delta, _ = policy.act(np.zeros(34, dtype=np.float32))
        np.testing.assert_allclose(delta, np.full(3, 0.004))


if __name__ == "__main__":
    unittest.main()
