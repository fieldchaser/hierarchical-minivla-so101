"""Phase-aware neural behavior cloning for the multicube task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .scripted_expert import PHASE_NAMES
from .state_policy import (
    ACTION_DELTA_LIMIT,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    load_episodes,
)

NUM_PHASES = len(PHASE_NAMES)


def phase_progress_summary(final_phases: Sequence[int]) -> dict[str, object]:
    """Summarize terminal phase progress for sparse-reward rollouts."""
    phases = np.asarray(final_phases, dtype=np.int64)
    if phases.size == 0:
        raise ValueError("At least one final phase is required")
    if np.any((phases < 0) | (phases >= NUM_PHASES)):
        raise ValueError("Final phase index is out of range")
    return {
        "mean_final_phase": float(phases.mean()),
        "final_phase_counts": {
            name: int(np.count_nonzero(phases == index))
            for index, name in enumerate(PHASE_NAMES)
        },
        "reached_phase_counts": {
            name: int(np.count_nonzero(phases >= index))
            for index, name in enumerate(PHASE_NAMES)
        },
    }


def observed_event_next_phase(
    observation: Mapping[str, np.ndarray],
    mocap_xyz: np.ndarray,
    current_phase: int,
    phase_steps: int,
) -> int:
    """Advance from task-completion events available in the state observation."""
    cubes = np.asarray(observation["cubes_xyz"]).reshape(3, 3)
    goal = np.asarray(observation["goal"])
    cube_xyz = goal @ cubes
    bin_xyz = np.asarray(observation["goal_pos"])
    gripper_angle = float(np.asarray(observation["gripper"]).reshape(-1)[0])
    grasp_target = np.array(
        [cube_xyz[0] - 0.015, cube_xyz[1] - 0.004, 0.078]
    )
    above_cube = grasp_target.copy()
    above_cube[2] = 0.15

    if current_phase == 0 and np.linalg.norm(mocap_xyz - above_cube) < 0.012:
        return 1
    if current_phase == 1 and np.linalg.norm(mocap_xyz - grasp_target) < 0.010:
        return 2
    if current_phase == 2 and phase_steps >= 20 and gripper_angle < 0.6:
        return 3
    if current_phase == 3 and cube_xyz[2] > 0.08:
        return 4
    if (
        current_phase == 4
        and cube_xyz[2] > 0.08
        and np.linalg.norm(cube_xyz[:2] - bin_xyz[:2]) < 0.05
    ):
        return 5
    return current_phase


def load_hierarchical_episodes(
    paths: Sequence[str | Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load flat-policy arrays plus the aligned expert phase labels."""
    features, deltas, gripper_open = load_episodes(paths)
    phases = []
    for path in paths:
        with np.load(path) as data:
            episode_phases = np.asarray(data["phase"], dtype=np.int64)
        phases.append(episode_phases)
    all_phases = np.concatenate(phases)
    if all_phases.shape[0] != features.shape[0]:
        raise ValueError("Phase labels are not aligned with the training arrays")
    return features, deltas, gripper_open, all_phases


def balanced_phase_weights(phases: np.ndarray) -> np.ndarray:
    """Return inverse-frequency class weights with mean weight one."""
    counts = np.bincount(np.asarray(phases), minlength=NUM_PHASES).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("Every phase must occur in the training split")
    weights = counts.sum() / (NUM_PHASES * counts)
    return weights / weights.mean()


@dataclass
class MonotonicPhaseTracker:
    """Advance one phase after repeated evidence and never move backward."""

    required_votes: int = 3
    current_phase: int = 0
    advance_votes: int = 0

    def update(self, predicted_phase: int) -> int:
        next_phase = self.current_phase + 1
        if predicted_phase == next_phase and next_phase < NUM_PHASES:
            self.advance_votes += 1
            if self.advance_votes >= self.required_votes:
                self.current_phase = next_phase
                self.advance_votes = 0
        else:
            self.advance_votes = 0
        return self.current_phase


class HierarchicalStateGoalPolicy(nn.Module):
    """Shared state encoder with a phase head and one action head per phase."""

    def __init__(
        self,
        feature_mean: np.ndarray | torch.Tensor,
        feature_std: np.ndarray | torch.Tensor,
        delta_mean: np.ndarray | torch.Tensor,
        delta_std: np.ndarray | torch.Tensor,
        phase_weights: np.ndarray | torch.Tensor,
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
        self.register_buffer("phase_weights", torch.as_tensor(phase_weights).float())
        input_dim = int(self.feature_mean.numel())
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(hidden_dim, NUM_PHASES)
        self.action_heads = nn.Linear(hidden_dim, NUM_PHASES * 4)

    def _outputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_std
        embedding = self.encoder(normalized)
        phase_logits = self.phase_head(embedding)
        actions = self.action_heads(embedding).reshape(-1, NUM_PHASES, 4)
        return phase_logits, actions

    def forward(
        self, features: torch.Tensor, phases: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_logits, actions = self._outputs(features)
        selected_phases = phase_logits.argmax(dim=1) if phases is None else phases
        rows = torch.arange(features.shape[0], device=features.device)
        selected_actions = actions[rows, selected_phases]
        delta = selected_actions[:, :3] * self.delta_std + self.delta_mean
        return delta, selected_actions[:, 3], phase_logits

    def loss(
        self,
        features: torch.Tensor,
        target_delta: torch.Tensor,
        target_gripper_open: torch.Tensor,
        target_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_logits, actions = self._outputs(features)
        rows = torch.arange(features.shape[0], device=features.device)
        selected_actions = actions[rows, target_phase]
        target_delta_normalized = (target_delta - self.delta_mean) / self.delta_std
        delta_loss = nn.functional.mse_loss(
            selected_actions[:, :3], target_delta_normalized
        )
        gripper_loss = nn.functional.binary_cross_entropy_with_logits(
            selected_actions[:, 3], target_gripper_open
        )
        phase_loss = nn.functional.cross_entropy(
            phase_logits, target_phase, weight=self.phase_weights
        )
        return (
            delta_loss + gripper_loss + phase_loss,
            delta_loss,
            gripper_loss,
            phase_loss,
        )

    @torch.no_grad()
    def predict_phase(self, feature: np.ndarray) -> int:
        device = next(self.parameters()).device
        features = torch.as_tensor(feature, dtype=torch.float32, device=device)[None]
        phase_logits, _ = self._outputs(features)
        return int(phase_logits.argmax(dim=1).item())

    @torch.no_grad()
    def act(self, feature: np.ndarray, phase: int | None = None) -> tuple[np.ndarray, float, int]:
        device = next(self.parameters()).device
        features = torch.as_tensor(feature, dtype=torch.float32, device=device)[None]
        predicted_phase = self.predict_phase(feature)
        selected_phase = predicted_phase if phase is None else phase
        phase_tensor = torch.tensor([selected_phase], dtype=torch.long, device=device)
        delta, gripper_logit, _ = self(features, phase_tensor)
        clipped_delta = delta[0].clamp(-ACTION_DELTA_LIMIT, ACTION_DELTA_LIMIT)
        gripper = GRIPPER_OPEN if gripper_logit.item() >= 0.0 else GRIPPER_CLOSED
        return clipped_delta.cpu().numpy(), gripper, predicted_phase


def save_hierarchical_policy(
    policy: HierarchicalStateGoalPolicy, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "hidden_dim": policy.hidden_dim,
            "phase_names": PHASE_NAMES,
            "state_dict": policy.state_dict(),
        },
        path,
    )
    return path


def load_hierarchical_policy(
    path: str | Path, device: str | torch.device = "cpu"
) -> HierarchicalStateGoalPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"]
    policy = HierarchicalStateGoalPolicy(
        state_dict["feature_mean"],
        state_dict["feature_std"],
        state_dict["delta_mean"],
        state_dict["delta_std"],
        state_dict["phase_weights"],
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    policy.load_state_dict(state_dict)
    return policy.to(device)
