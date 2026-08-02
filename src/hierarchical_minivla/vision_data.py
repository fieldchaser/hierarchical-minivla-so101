"""Validation helpers for frame-aligned vision-language episodes."""

from __future__ import annotations

from typing import Mapping

import numpy as np

VISION_TEMPORAL_KEYS = (
    "rgb",
    "instruction",
    "state_ee_xyz",
    "state_mocap_xyz",
    "state_joints",
    "state_gripper",
    "cubes_xyz",
    "state_goal",
    "goal_pos",
    "action_ee_xyz",
    "action_gripper",
    "phase",
)


def validate_vision_episode_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Validate the temporal contract of one RGB-language-action episode."""
    missing = [key for key in VISION_TEMPORAL_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Missing vision episode arrays: {missing}")

    lengths = {key: len(arrays[key]) for key in VISION_TEMPORAL_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Vision episode arrays are not aligned: {lengths}")

    rgb = np.asarray(arrays["rgb"])
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be uint8 with shape T x H x W x 3")
    instructions = np.asarray(arrays["instruction"])
    if instructions.ndim != 1 or instructions.dtype.kind not in ("U", "S"):
        raise ValueError("instruction must be a one-dimensional string array")
    unique_instructions = np.unique(instructions.astype(str))
    if unique_instructions.size != 1:
        raise ValueError("One episode must contain exactly one instruction")

    return {
        "num_steps": int(rgb.shape[0]),
        "image_shape": list(rgb.shape[1:]),
        "image_dtype": str(rgb.dtype),
        "instruction": str(unique_instructions[0]),
    }
