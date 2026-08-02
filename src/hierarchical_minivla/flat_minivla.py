"""A small RGB-language-proprioception policy for manipulation."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .state_policy import ACTION_DELTA_LIMIT, GRIPPER_CLOSED, GRIPPER_OPEN
from .vision_data import validate_vision_episode_arrays

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PROPRIO_DIM = 13
MOVEMENT_THRESHOLD = 0.0005
GRASP_ACTIVE_THRESHOLD = 0.35


def instruction_tokens(instruction: str) -> list[str]:
    """Lowercase and split an instruction into simple word tokens."""
    return re.findall(r"[a-z0-9]+", instruction.lower())


def build_vocabulary(instructions: Sequence[str]) -> dict[str, int]:
    """Build a deterministic word vocabulary with padding and unknown tokens."""
    words = sorted({word for text in instructions for word in instruction_tokens(text)})
    return {PAD_TOKEN: 0, UNK_TOKEN: 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode_instructions(
    instructions: Sequence[str], vocabulary: dict[str, int]
) -> np.ndarray:
    """Encode and right-pad instructions using a fixed vocabulary."""
    tokenized = [instruction_tokens(text) for text in instructions]
    max_tokens = max(len(tokens) for tokens in tokenized)
    encoded = np.zeros((len(tokenized), max_tokens), dtype=np.int64)
    unknown = vocabulary[UNK_TOKEN]
    for row, tokens in enumerate(tokenized):
        encoded[row, : len(tokens)] = [vocabulary.get(token, unknown) for token in tokens]
    return encoded


def load_vision_episodes(paths: Sequence[str | Path]) -> dict[str, np.ndarray]:
    """Load successful RGB episodes into aligned multimodal training arrays."""
    collected: dict[str, list[np.ndarray]] = {
        "rgb": [],
        "proprio": [],
        "instruction": [],
        "delta": [],
        "gripper_open": [],
        "phase": [],
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            validate_vision_episode_arrays(data)
            if not bool(data["success"]):
                raise ValueError(f"Vision episode is not successful: {path}")
            proprio = np.concatenate(
                [
                    data["state_ee_xyz"],
                    data["state_mocap_xyz"],
                    data["state_joints"],
                    data["state_gripper"],
                ],
                axis=1,
            ).astype(np.float32)
            if proprio.shape[1] != PROPRIO_DIM:
                raise ValueError(f"Expected {PROPRIO_DIM} proprioceptive values")
            collected["rgb"].append(np.asarray(data["rgb"], dtype=np.uint8))
            collected["proprio"].append(proprio)
            collected["instruction"].append(np.asarray(data["instruction"]).astype(str))
            collected["delta"].append(
                np.asarray(data["action_ee_xyz"], dtype=np.float32)
            )
            gripper = np.asarray(data["action_gripper"], dtype=np.float32)[:, 0]
            collected["gripper_open"].append((gripper > 0.4).astype(np.float32))
            collected["phase"].append(np.asarray(data["phase"], dtype=np.int64))
    if not collected["rgb"]:
        raise ValueError("No vision episodes were provided")
    return {key: np.concatenate(values) for key, values in collected.items()}


def proprio_from_observation(
    observation: Mapping[str, np.ndarray], command_xyz: np.ndarray
) -> np.ndarray:
    """Build the same robot-only proprioceptive vector used during training."""
    proprio = np.concatenate(
        [
            observation["ee_pos"],
            np.asarray(command_xyz),
            observation["joints"],
            observation["gripper"],
        ]
    ).astype(np.float32)
    if proprio.shape != (PROPRIO_DIM,):
        raise ValueError(f"Expected proprioception shape ({PROPRIO_DIM},)")
    return proprio


def transition_focused_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Remove repeated waits that give a memoryless policy conflicting labels."""
    phases = np.asarray(arrays["phase"])
    deltas = np.asarray(arrays["delta"])
    proprio = np.asarray(arrays["proprio"])
    moving = np.linalg.norm(deltas, axis=1) > MOVEMENT_THRESHOLD
    active_grasp = (phases == 2) & (
        proprio[:, -1] > GRASP_ACTIVE_THRESHOLD
    )
    release = phases == 5
    return moving | active_grasp | release


def split_vision_episode_paths(
    paths: Sequence[str | Path], seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Hold out one whole episode for every distinct training instruction."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as data:
            instructions = np.unique(np.asarray(data["instruction"]).astype(str))
        if len(instructions) != 1:
            raise ValueError(f"Expected one instruction in episode: {path}")
        groups[str(instructions[0])].append(path)
    if not groups:
        raise ValueError("No vision episodes were provided")

    rng = np.random.default_rng(seed)
    train_paths = []
    validation_paths = []
    for instruction in sorted(groups):
        group = sorted(groups[instruction])
        if len(group) < 2:
            raise ValueError(
                f"Need at least two episodes for instruction: {instruction}"
            )
        validation_index = int(rng.integers(len(group)))
        validation_paths.append(group[validation_index])
        train_paths.extend(
            path for index, path in enumerate(group) if index != validation_index
        )
    return sorted(train_paths), sorted(validation_paths)


def phase_balanced_indices(phases: np.ndarray, batch_size: int) -> np.ndarray:
    """Select evenly spaced examples with near-equal coverage of present phases."""
    phase_ids = np.unique(np.asarray(phases, dtype=np.int64))
    base, remainder = divmod(batch_size, len(phase_ids))
    selected = []
    for position, phase in enumerate(phase_ids):
        count = base + int(position < remainder)
        candidates = np.flatnonzero(phases == phase)
        offsets = np.linspace(0, len(candidates) - 1, count, dtype=int)
        selected.append(candidates[offsets])
    return np.concatenate(selected)


class FlatMiniVLA(nn.Module):
    """Fuse a small CNN, word embeddings, and robot proprioception."""

    def __init__(
        self,
        vocab_size: int,
        proprio_mean: np.ndarray | torch.Tensor,
        proprio_std: np.ndarray | torch.Tensor,
        delta_mean: np.ndarray | torch.Tensor,
        delta_std: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("proprio_mean", torch.as_tensor(proprio_mean).float())
        self.register_buffer(
            "proprio_std", torch.as_tensor(proprio_std).float().clamp_min(1e-6)
        )
        self.register_buffer("delta_mean", torch.as_tensor(delta_mean).float())
        self.register_buffer(
            "delta_std", torch.as_tensor(delta_std).float().clamp_min(1e-6)
        )
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
        )
        self.text_embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.proprio_encoder = nn.Sequential(nn.Linear(PROPRIO_DIM, 32), nn.ReLU())
        self.action_head = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(
        self, rgb: torch.Tensor, tokens: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.encode_observation(rgb, tokens, proprio)
        output = self.action_head(fused)
        delta = output[:, :3] * self.delta_std + self.delta_mean
        return delta, output[:, 3]

    def encode_observation(
        self, rgb: torch.Tensor, tokens: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        """Encode and concatenate vision, language, and robot state."""
        images = rgb.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        vision = self.vision_encoder(images)
        token_mask = (tokens != 0).unsqueeze(-1)
        embedded = self.text_embedding(tokens)
        language = (embedded * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        normalized_proprio = (proprio - self.proprio_mean) / self.proprio_std
        robot = self.proprio_encoder(normalized_proprio)
        return torch.cat([vision, language, robot], dim=1)

    def loss(
        self,
        rgb: torch.Tensor,
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        target_delta: torch.Tensor,
        target_gripper_open: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_delta, gripper_logit = self(rgb, tokens, proprio)
        normalized_prediction = (predicted_delta - self.delta_mean) / self.delta_std
        normalized_target = (target_delta - self.delta_mean) / self.delta_std
        delta_loss = nn.functional.mse_loss(normalized_prediction, normalized_target)
        gripper_loss = nn.functional.binary_cross_entropy_with_logits(
            gripper_logit, target_gripper_open
        )
        return delta_loss + gripper_loss, delta_loss, gripper_loss

    @torch.inference_mode()
    def act(
        self, rgb: np.ndarray, tokens: np.ndarray, proprio: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Predict one bounded Cartesian delta and a simulator gripper command."""
        device = next(self.parameters()).device
        rgb_tensor = torch.as_tensor(rgb, device=device).unsqueeze(0)
        token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
        proprio_tensor = torch.as_tensor(
            proprio, dtype=torch.float32, device=device
        ).unsqueeze(0)
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        delta, gripper_logit = self(rgb_tensor, token_tensor, proprio_tensor)
        bounded_delta = delta[0].clamp(-ACTION_DELTA_LIMIT, ACTION_DELTA_LIMIT)
        gripper = GRIPPER_OPEN if gripper_logit.item() >= 0.0 else GRIPPER_CLOSED
        return bounded_delta.cpu().numpy(), gripper


def save_flat_minivla(
    policy: FlatMiniVLA, vocabulary: dict[str, int], path: str | Path
) -> Path:
    """Save a Flat MiniVLA checkpoint and its word vocabulary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"version": 3, "vocabulary": vocabulary, "state_dict": policy.state_dict()},
        path,
    )
    return path


def load_flat_minivla(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[FlatMiniVLA, dict[str, int]]:
    """Restore a Flat MiniVLA checkpoint and its word vocabulary."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("version") != 3:
        raise ValueError(f"Unsupported Flat MiniVLA version: {checkpoint.get('version')}")
    state_dict = checkpoint["state_dict"]
    vocabulary = checkpoint["vocabulary"]
    policy = FlatMiniVLA(
        vocab_size=len(vocabulary),
        proprio_mean=state_dict["proprio_mean"],
        proprio_std=state_dict["proprio_std"],
        delta_mean=state_dict["delta_mean"],
        delta_std=state_dict["delta_std"],
    )
    policy.load_state_dict(state_dict)
    return policy.to(map_location), vocabulary
