"""Neural state-and-goal behavior cloning for the multicube task."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

FEATURE_KEYS = (
    "state_ee_xyz",
    "state_joints",
    "state_gripper",
    "cubes_xyz",
    "state_goal",
    "goal_pos",
)
ACTION_DELTA_LIMIT = 0.004
GRIPPER_CLOSED = -0.174
GRIPPER_OPEN = 1.0


def build_features(
    ee_xyz: np.ndarray,
    joints: np.ndarray,
    gripper: np.ndarray,
    cubes_xyz: np.ndarray,
    goal: np.ndarray,
    bin_xyz: np.ndarray,
) -> np.ndarray:
    """Add explicit goal selection and relative geometry to the raw state."""
    cubes = np.asarray(cubes_xyz).reshape(-1, 3, 3)
    selected_cube = np.einsum("nij,ni->nj", cubes, goal)
    return np.concatenate(
        [
            ee_xyz,
            joints,
            gripper,
            cubes_xyz,
            goal,
            bin_xyz,
            selected_cube,
            selected_cube - ee_xyz,
            bin_xyz - ee_xyz,
        ],
        axis=1,
    ).astype(np.float32)


def load_episodes(paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load successful episode archives into aligned training arrays."""
    features = []
    deltas = []
    gripper_open = []
    for path in paths:
        with np.load(path) as data:
            if not bool(data["success"]):
                raise ValueError(f"Training episode is not successful: {path}")
            states = {
                key: np.asarray(data[key], dtype=np.float32) for key in FEATURE_KEYS
            }
            episode_features = build_features(
                states["state_ee_xyz"],
                states["state_joints"],
                states["state_gripper"],
                states["cubes_xyz"],
                states["state_goal"],
                states["goal_pos"],
            )
            episode_deltas = np.asarray(data["action_ee_xyz"], dtype=np.float32)
            episode_gripper = np.asarray(data["action_gripper"], dtype=np.float32)

        lengths = {
            episode_features.shape[0],
            episode_deltas.shape[0],
            episode_gripper.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError(f"Temporal arrays are not aligned in {path}")
        features.append(episode_features)
        deltas.append(episode_deltas)
        gripper_open.append((episode_gripper[:, 0] > 0.4).astype(np.float32))

    if not features:
        raise ValueError("No episode files were provided")
    return (
        np.concatenate(features),
        np.concatenate(deltas),
        np.concatenate(gripper_open),
    )


def split_episode_paths(
    paths: Sequence[Path], validation_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Split whole episodes so adjacent timesteps cannot leak across splits."""
    if len(paths) < 2:
        raise ValueError("At least two episodes are required for a train/validation split")
    rng = np.random.default_rng(seed)
    shuffled = [paths[index] for index in rng.permutation(len(paths))]
    num_validation = max(1, round(len(paths) * validation_fraction))
    num_validation = min(num_validation, len(paths) - 1)
    return shuffled[num_validation:], shuffled[:num_validation]


def observation_to_feature(observation: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert one upstream environment observation to the learned policy input."""
    return build_features(
        observation["ee_pos"][None],
        observation["joints"][None],
        observation["gripper"][None],
        observation["cubes_xyz"][None],
        observation["goal"][None],
        observation["goal_pos"][None],
    )[0]


class StateGoalPolicy(nn.Module):
    """A small MLP with Cartesian-regression and binary-gripper outputs."""

    def __init__(
        self,
        feature_mean: np.ndarray | torch.Tensor,
        feature_std: np.ndarray | torch.Tensor,
        delta_mean: np.ndarray | torch.Tensor,
        delta_std: np.ndarray | torch.Tensor,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean).float())
        self.register_buffer(
            "feature_std", torch.as_tensor(feature_std).float().clamp_min(1e-6)
        )
        self.register_buffer("delta_mean", torch.as_tensor(delta_mean).float())
        self.register_buffer(
            "delta_std", torch.as_tensor(delta_std).float().clamp_min(1e-6)
        )
        input_dim = int(self.feature_mean.numel())
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def _raw_output(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.network(normalized)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._raw_output(features)
        delta = raw[:, :3] * self.delta_std + self.delta_mean
        return delta, raw[:, 3]

    def loss(
        self,
        features: torch.Tensor,
        target_delta: torch.Tensor,
        target_gripper_open: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self._raw_output(features)
        target_delta_normalized = (target_delta - self.delta_mean) / self.delta_std
        delta_loss = nn.functional.mse_loss(raw[:, :3], target_delta_normalized)
        gripper_loss = nn.functional.binary_cross_entropy_with_logits(
            raw[:, 3], target_gripper_open
        )
        return delta_loss + gripper_loss, delta_loss, gripper_loss

    @torch.no_grad()
    def act(self, feature: np.ndarray) -> tuple[np.ndarray, float]:
        device = next(self.parameters()).device
        features = torch.as_tensor(feature, dtype=torch.float32, device=device)[None]
        delta, gripper_logit = self(features)
        clipped_delta = delta[0].clamp(-ACTION_DELTA_LIMIT, ACTION_DELTA_LIMIT)
        gripper = GRIPPER_OPEN if gripper_logit.item() >= 0.0 else GRIPPER_CLOSED
        return clipped_delta.cpu().numpy(), gripper


def save_policy(policy: StateGoalPolicy, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "hidden_dim": policy.hidden_dim,
            "state_dict": policy.state_dict(),
        },
        path,
    )
    return path


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> StateGoalPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"]
    policy = StateGoalPolicy(
        state_dict["feature_mean"],
        state_dict["feature_std"],
        state_dict["delta_mean"],
        state_dict["delta_std"],
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    policy.load_state_dict(state_dict)
    return policy.to(device)
