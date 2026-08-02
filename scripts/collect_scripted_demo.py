#!/usr/bin/env python3
"""Collect one fixed-layout red-cube demonstration from the ETH MuJoCo scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco

from hierarchical_minivla.scripted_expert import run_scripted_episode, save_scripted_episode


class PhysicsOnlyRenderer:
    """Avoid creating an OpenGL context during headless data collection."""

    def __init__(self, *args, **kwargs) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-hw3", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scripted/red_fixed_seed0.npz"),
    )
    args = parser.parse_args()

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")

    mujoco.Renderer = PhysicsOnlyRenderer
    sys.path.insert(0, str(hw3_path))
    from hw3.sim_env import SO100MulticubeSimEnv

    env = SO100MulticubeSimEnv(
        xml_path=xml_path,
        goal_cube="red",
        cube_pos_std=0.0,
        shuffle_cubes=False,
        seed=0,
    )
    episode = run_scripted_episode(env)
    output_path = save_scripted_episode(episode, args.output.resolve())

    phase_ids = episode.arrays["phase"]
    phase_names = episode.arrays["phase_names"]
    phase_steps = {
        str(name): int((phase_ids == index).sum())
        for index, name in enumerate(phase_names)
    }
    print(
        json.dumps(
            {
                "success": episode.success,
                "num_steps": episode.num_steps,
                "phase_steps": phase_steps,
                "grasp_contact_angle": round(episode.grasp_contact_angle, 4),
                "final_cube_xyz": episode.final_cube_xyz.round(4).tolist(),
                "bin_xyz": episode.bin_xyz.round(4).tolist(),
                "output": str(output_path),
            },
            indent=2,
        )
    )
    if not episode.success:
        raise SystemExit("Scripted expert failed to release the red cube in the bin")


if __name__ == "__main__":
    main()
