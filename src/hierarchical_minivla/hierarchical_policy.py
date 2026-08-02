"""Phase-aware neural behavior cloning for the multicube task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
