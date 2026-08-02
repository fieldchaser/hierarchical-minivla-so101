"""A fixed-layout MuJoCo expert that produces one real pick-and-place trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PHASE_NAMES = (
    "reach",
    "descend",
    "grasp",
    "lift",
    "transport",
    "release",
)


@dataclass(frozen=True)
class ScriptedEpisode:
    success: bool
    arrays: dict[str, np.ndarray]
    final_cube_xyz: np.ndarray
    bin_xyz: np.ndarray
    grasp_contact_angle: float

    @property
    def num_steps(self) -> int:
        return int(self.arrays["action_ee_xyz"].shape[0])


def bounded_delta(current: np.ndarray, target: np.ndarray, limit: float = 0.004) -> np.ndarray:
    """Return one bounded Cartesian control increment toward ``target``."""
    return np.clip(np.asarray(target) - np.asarray(current), -limit, limit)


def is_released_in_bin(
    cube_xyz: np.ndarray,
    bin_xyz: np.ndarray,
    gripper_angle: float,
    xy_threshold: float = 0.04,
) -> bool:
    """Require the target cube to be in the bin while the gripper is open."""
    cube_xyz = np.asarray(cube_xyz)
    bin_xyz = np.asarray(bin_xyz)
    in_xy = np.all(np.abs(cube_xyz[:2] - bin_xyz[:2]) < xy_threshold)
    in_z = 0.0 < cube_xyz[2] < 0.04
    return bool(in_xy and in_z and gripper_angle > 0.5)


def run_scripted_episode(env: Any) -> ScriptedEpisode:
    """Run a deterministic red-cube pick-and-place episode.

    The environment must be the fixed-layout upstream multicube scene with the
    red cube selected as its goal. States are captured before each command so
    every row is a behavior-cloning pair ``(observation_t, action_t)``.
    """
    records: dict[str, list[np.ndarray | int | float]] = {
        "state_ee_xyz": [],
        "state_joints": [],
        "state_gripper": [],
        "cubes_xyz": [],
        "cubes": [],
        "state_goal": [],
        "goal_pos": [],
        "action_ee_xyz": [],
        "action_gripper": [],
        "phase": [],
    }

    def step(target: np.ndarray, gripper_command: float, phase: str) -> None:
        obs = env.get_obs()
        mocap = env.data.mocap_pos[env.mocap_id].copy()
        delta = bounded_delta(mocap, target)

        records["state_ee_xyz"].append(obs["ee_pos"].copy())
        records["state_joints"].append(obs["joints"].copy())
        records["state_gripper"].append(obs["gripper"].copy())
        records["cubes_xyz"].append(obs["cubes_xyz"].copy())
        records["cubes"].append(obs["cubes"].copy())
        records["state_goal"].append(obs["goal"].copy())
        records["goal_pos"].append(obs["goal_pos"].copy())
        records["action_ee_xyz"].append(delta.copy())
        records["action_gripper"].append(float(gripper_command))
        records["phase"].append(PHASE_NAMES.index(phase))

        env.set_mocap_pos(mocap + delta)
        env.set_gripper(gripper_command)
        env.step()

    def move_to(
        target: np.ndarray,
        gripper_command: float,
        phase: str,
        settle_steps: int,
        max_steps: int = 120,
    ) -> None:
        for _ in range(max_steps):
            mocap = env.data.mocap_pos[env.mocap_id]
            if np.linalg.norm(np.asarray(target) - mocap) < 0.001:
                break
            step(target, gripper_command, phase)
        else:
            raise RuntimeError(f"Expert phase {phase!r} did not reach its target")

        for _ in range(settle_steps):
            step(env.data.mocap_pos[env.mocap_id].copy(), gripper_command, phase)

    def hold(gripper_command: float, phase: str, num_steps: int) -> None:
        for _ in range(num_steps):
            step(env.data.mocap_pos[env.mocap_id].copy(), gripper_command, phase)

    cube_xyz = env.get_target_cube_state()[:3].copy()
    bin_xyz = env.get_goal_pos().copy()

    # The ee_site is at the wrist, not between the fingertips. This measured
    # offset aligns the lower jaw pads around the 4 cm cube in the ETH scene.
    grasp_target = np.array(
        [cube_xyz[0] - 0.015, cube_xyz[1] - 0.004, 0.078], dtype=np.float64
    )
    above_cube = grasp_target.copy()
    above_cube[2] = 0.15

    move_to(above_cube, gripper_command=1.0, phase="reach", settle_steps=5)
    move_to(grasp_target, gripper_command=1.0, phase="descend", settle_steps=8)
    hold(gripper_command=-0.174, phase="grasp", num_steps=30)
    grasp_contact_angle = env.get_gripper_angle()

    lift_target = grasp_target.copy()
    lift_target[2] = 0.17
    move_to(lift_target, gripper_command=-0.174, phase="lift", settle_steps=8)

    # Preserve the measured wrist-to-cube offset so the cube, rather than the
    # wrist site, is positioned over the center of the bin.
    held_offset = env.data.mocap_pos[env.mocap_id] - env.get_target_cube_state()[:3]
    transport_target = np.array(
        [bin_xyz[0] + held_offset[0], bin_xyz[1] + held_offset[1], 0.17],
        dtype=np.float64,
    )
    move_to(transport_target, gripper_command=-0.174, phase="transport", settle_steps=10)
    hold(gripper_command=1.0, phase="release", num_steps=45)

    final_cube_xyz = env.get_target_cube_state()[:3].copy()
    success = is_released_in_bin(
        final_cube_xyz,
        bin_xyz,
        env.get_gripper_angle(),
    )

    arrays = {
        key: np.asarray(values, dtype=np.int64 if key == "phase" else np.float32)
        for key, values in records.items()
    }
    arrays["action_gripper"] = arrays["action_gripper"].reshape(-1, 1)
    arrays["phase_names"] = np.asarray(PHASE_NAMES)
    return ScriptedEpisode(
        success,
        arrays,
        final_cube_xyz,
        bin_xyz,
        grasp_contact_angle,
    )


def save_scripted_episode(episode: ScriptedEpisode, path: str | Path) -> Path:
    """Save a scripted episode as a compressed NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **episode.arrays,
        success=np.asarray(episode.success),
        final_cube_xyz=episode.final_cube_xyz.astype(np.float32),
        bin_xyz=episode.bin_xyz.astype(np.float32),
        grasp_contact_angle=np.asarray(episode.grasp_contact_angle, dtype=np.float32),
    )
    return path
