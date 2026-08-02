"""A minimal goal-conditioned behavior-cloning baseline.

The baseline is deliberately small. It validates the central supervised-learning
contract before we introduce images, language encoders, or action chunking:

    robot state + scene state + task goal -> expert action
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE_DIM = 19
ACTION_DIM = 4


def _as_batch(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state[None, :]
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"Expected state shape (N, {STATE_DIM}), got {state.shape}")
    return state


def select_goal_cube(state: np.ndarray) -> np.ndarray:
    """Select the target cube position using the one-hot goal."""
    state = _as_batch(state)
    cubes_xyz = state[:, 4:13].reshape(-1, 3, 3)
    goal_onehot = state[:, 13:16]
    return np.einsum("nij,ni->nj", cubes_xyz, goal_onehot)


def featurize(state: np.ndarray) -> np.ndarray:
    """Add the multiplicative interaction needed for goal conditioning.

    A plain linear model cannot express "use the red position when the red goal
    bit is on" because that requires multiplying goal bits by cube positions.
    The selected target and motion-target features make that interaction explicit.
    """
    state = _as_batch(state)
    ee_xyz = state[:, :3]
    holding = state[:, 3:4]
    selected_cube = select_goal_cube(state)
    bin_xyz = state[:, 16:19]
    motion_target = (1.0 - holding) * selected_cube + holding * bin_xyz
    return np.concatenate(
        [state, selected_cube, selected_cube - ee_xyz, bin_xyz - ee_xyz, motion_target - ee_xyz],
        axis=1,
    )


def expert_action(state: np.ndarray) -> np.ndarray:
    """Return a tiny scripted expert action for synthetic smoke-test states.

    The first three values are an end-effector position delta. The final value
    is an absolute gripper command in {-1, +1}. This is not a physics expert;
    it only creates a deterministic target for testing the BC training path.
    """
    state = _as_batch(state)
    ee_xyz = state[:, :3]
    holding = state[:, 3:4]
    selected_cube = select_goal_cube(state)
    bin_xyz = state[:, 16:19]
    motion_target = (1.0 - holding) * selected_cube + holding * bin_xyz
    delta_xyz = np.clip(motion_target - ee_xyz, -0.04, 0.04)
    gripper = 2.0 * holding - 1.0
    return np.concatenate([delta_xyz, gripper], axis=1)


def make_dataset(num_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic synthetic demonstrations for the smoke test."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    rng = np.random.default_rng(seed)
    cubes_xyz = rng.uniform([-0.25, 0.45, 0.02], [0.25, 0.75, 0.02], size=(num_samples, 3, 3))
    bin_xyz = rng.uniform([-0.2, 0.45, 0.03], [0.2, 0.75, 0.03], size=(num_samples, 3))
    goal_index = rng.integers(0, 3, size=num_samples)
    goal_onehot = np.eye(3, dtype=np.float64)[goal_index]
    holding = rng.integers(0, 2, size=(num_samples, 1)).astype(np.float64)

    selected_cube = np.einsum("nij,ni->nj", cubes_xyz, goal_onehot)
    motion_target = (1.0 - holding) * selected_cube + holding * bin_xyz
    ee_xyz = motion_target + rng.uniform(-0.035, 0.035, size=(num_samples, 3))

    states = np.concatenate(
        [ee_xyz, holding, cubes_xyz.reshape(num_samples, -1), goal_onehot, bin_xyz],
        axis=1,
    )
    return states, expert_action(states)


@dataclass
class GoalConditionedRidgePolicy:
    """Closed-form ridge regression over goal-conditioned features."""

    regularization: float = 1e-6
    weights: np.ndarray | None = None

    def fit(self, states: np.ndarray, actions: np.ndarray) -> "GoalConditionedRidgePolicy":
        features = featurize(states)
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (features.shape[0], ACTION_DIM):
            raise ValueError(
                f"Expected action shape ({features.shape[0]}, {ACTION_DIM}), got {actions.shape}"
            )

        design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
        penalty = np.eye(design.shape[1]) * self.regularization
        penalty[-1, -1] = 0.0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ actions)
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit() or load() before predict().")
        features = featurize(states)
        design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
        predictions = design @ self.weights
        predictions[:, :3] = np.clip(predictions[:, :3], -0.04, 0.04)
        predictions[:, 3] = np.clip(predictions[:, 3], -1.0, 1.0)
        return predictions

    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("Cannot save an unfitted policy.")
        np.savez(path, weights=self.weights, regularization=self.regularization)

    @classmethod
    def load(cls, path: str | Path) -> "GoalConditionedRidgePolicy":
        with np.load(path) as payload:
            return cls(
                regularization=float(payload["regularization"]),
                weights=np.asarray(payload["weights"], dtype=np.float64),
            )
