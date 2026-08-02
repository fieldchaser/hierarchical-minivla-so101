"""Phase-aware visual-language-action policy for SO-101 manipulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .flat_minivla import FlatMiniVLA
from .scripted_expert import PHASE_NAMES
from .state_policy import ACTION_DELTA_LIMIT, GRIPPER_CLOSED, GRIPPER_OPEN

NUM_PHASES = len(PHASE_NAMES)
FUSED_DIM = 128


class HierarchicalMiniVLA(FlatMiniVLA):
    """Shared multimodal encoder with phase-specific action prediction."""

    def __init__(
        self,
        vocab_size: int,
        proprio_mean: np.ndarray | torch.Tensor,
        proprio_std: np.ndarray | torch.Tensor,
        delta_mean: np.ndarray | torch.Tensor,
        delta_std: np.ndarray | torch.Tensor,
        phase_weights: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__(
            vocab_size,
            proprio_mean,
            proprio_std,
            delta_mean,
            delta_std,
        )
        del self.action_head
        self.register_buffer("phase_weights", torch.as_tensor(phase_weights).float())
        if self.phase_weights.shape != (NUM_PHASES,):
            raise ValueError(f"Expected {NUM_PHASES} phase weights")
        self.fusion_encoder = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(FUSED_DIM, NUM_PHASES)
        self.action_heads = nn.Linear(FUSED_DIM, NUM_PHASES * 4)

    def _outputs(
        self, rgb: torch.Tensor, tokens: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion_encoder(self.encode_observation(rgb, tokens, proprio))
        phase_logits = self.phase_head(fused)
        actions = self.action_heads(fused).reshape(-1, NUM_PHASES, 4)
        return phase_logits, actions

    def forward(
        self,
        rgb: torch.Tensor,
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        phases: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_logits, actions = self._outputs(rgb, tokens, proprio)
        selected_phases = phase_logits.argmax(dim=1) if phases is None else phases
        rows = torch.arange(len(rgb), device=rgb.device)
        selected_actions = actions[rows, selected_phases]
        delta = selected_actions[:, :3] * self.delta_std + self.delta_mean
        return delta, selected_actions[:, 3], phase_logits

    def loss(
        self,
        rgb: torch.Tensor,
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        target_delta: torch.Tensor,
        target_gripper_open: torch.Tensor,
        target_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_delta, gripper_logit, phase_logits = self(
            rgb, tokens, proprio, target_phase
        )
        normalized_prediction = (predicted_delta - self.delta_mean) / self.delta_std
        normalized_target = (target_delta - self.delta_mean) / self.delta_std
        delta_loss = nn.functional.mse_loss(
            normalized_prediction, normalized_target
        )
        gripper_loss = nn.functional.binary_cross_entropy_with_logits(
            gripper_logit, target_gripper_open
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

    @torch.inference_mode()
    def act(
        self,
        rgb: np.ndarray,
        tokens: np.ndarray,
        proprio: np.ndarray,
        phase: int | None = None,
    ) -> tuple[np.ndarray, float, int]:
        device = next(self.parameters()).device
        rgb_tensor = torch.as_tensor(rgb, device=device).unsqueeze(0)
        token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
        proprio_tensor = torch.as_tensor(
            proprio, dtype=torch.float32, device=device
        ).unsqueeze(0)
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        phase_logits, _ = self._outputs(rgb_tensor, token_tensor, proprio_tensor)
        predicted_phase = int(phase_logits.argmax(dim=1).item())
        selected_phase = predicted_phase if phase is None else phase
        phase_tensor = torch.tensor([selected_phase], device=device)
        delta, gripper_logit, _ = self(
            rgb_tensor, token_tensor, proprio_tensor, phase_tensor
        )
        bounded_delta = delta[0].clamp(-ACTION_DELTA_LIMIT, ACTION_DELTA_LIMIT)
        gripper = GRIPPER_OPEN if gripper_logit.item() >= 0.0 else GRIPPER_CLOSED
        return bounded_delta.cpu().numpy(), gripper, predicted_phase


def save_hierarchical_minivla(
    policy: HierarchicalMiniVLA, vocabulary: dict[str, int], path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "phase_names": PHASE_NAMES,
            "vocabulary": vocabulary,
            "state_dict": policy.state_dict(),
        },
        path,
    )
    return path


def load_hierarchical_minivla(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[HierarchicalMiniVLA, dict[str, int]]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError(
            f"Unsupported Hierarchical MiniVLA version: {checkpoint.get('version')}"
        )
    if tuple(checkpoint["phase_names"]) != PHASE_NAMES:
        raise ValueError("Checkpoint phase names do not match the project contract")
    state_dict = checkpoint["state_dict"]
    vocabulary = checkpoint["vocabulary"]
    policy = HierarchicalMiniVLA(
        vocab_size=len(vocabulary),
        proprio_mean=state_dict["proprio_mean"],
        proprio_std=state_dict["proprio_std"],
        delta_mean=state_dict["delta_mean"],
        delta_std=state_dict["delta_std"],
        phase_weights=state_dict["phase_weights"],
    )
    policy.load_state_dict(state_dict)
    return policy.to(map_location), vocabulary
