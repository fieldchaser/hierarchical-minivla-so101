from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hierarchical_minivla.baseline import (
    GoalConditionedRidgePolicy,
    expert_action,
    make_dataset,
    select_goal_cube,
)


class BaselineTest(unittest.TestCase):
    def test_goal_selects_the_requested_cube(self) -> None:
        states, _ = make_dataset(1, seed=3)
        cubes = states[0, 4:13].reshape(3, 3)
        for goal_index in range(3):
            states[0, 13:16] = np.eye(3)[goal_index]
            np.testing.assert_allclose(select_goal_cube(states)[0], cubes[goal_index])

    def test_policy_fits_the_scripted_expert(self) -> None:
        train_states, train_actions = make_dataset(1_000, seed=4)
        test_states, test_actions = make_dataset(200, seed=5)
        policy = GoalConditionedRidgePolicy().fit(train_states, train_actions)
        mse = np.mean((policy.predict(test_states) - test_actions) ** 2)
        self.assertLess(mse, 1e-8)

    def test_save_load_round_trip(self) -> None:
        states, actions = make_dataset(100, seed=6)
        policy = GoalConditionedRidgePolicy().fit(states, actions)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy.npz"
            policy.save(path)
            loaded = GoalConditionedRidgePolicy.load(path)
            np.testing.assert_allclose(loaded.predict(states), policy.predict(states))

    def test_expert_action_has_expected_shape(self) -> None:
        states, _ = make_dataset(11, seed=9)
        self.assertEqual(expert_action(states).shape, (11, 4))

    def test_policy_outputs_respect_action_bounds(self) -> None:
        states, actions = make_dataset(100, seed=10)
        policy = GoalConditionedRidgePolicy().fit(states, actions)
        probe = states[:3].copy()
        probe[:, :3] += 10.0
        predictions = policy.predict(probe)
        self.assertTrue(np.all(np.abs(predictions[:, :3]) <= 0.04))
        self.assertTrue(np.all(np.abs(predictions[:, 3]) <= 1.0))


if __name__ == "__main__":
    unittest.main()
