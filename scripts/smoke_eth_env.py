#!/usr/bin/env python3
"""Instantiate and step the upstream ETH HW3 multicube environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eth-hw3",
        type=Path,
        required=True,
        help="Path to the upstream ethz-course-2026/hw3_imitation_learning directory.",
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")

    sys.path.insert(0, str(hw3_path))
    if not args.render:
        import mujoco

        class PhysicsOnlyRenderer:
            """Avoid creating an OpenGL context during a headless physics check."""

            def __init__(self, *args, **kwargs) -> None:
                pass

        mujoco.Renderer = PhysicsOnlyRenderer

    from hw3.sim_env import SO100MulticubeSimEnv

    env = SO100MulticubeSimEnv(xml_path=xml_path, seed=0)
    first_obs = env.get_obs()
    second_obs = env.step()
    summary = {
        "scene": str(xml_path),
        "goal_onehot": second_obs["goal"].tolist(),
        "joint_shape": list(second_obs["joints"].shape),
        "cubes_xyz_shape": list(second_obs["cubes_xyz"].shape),
        "bin_xyz": second_obs["goal_pos"].round(4).tolist(),
        "sim_time_advanced": bool(env.data.time > 0),
        "cube_positions_changed_one_step": bool(
            (first_obs["cubes_xyz"] != second_obs["cubes_xyz"]).any()
        ),
    }
    if args.render:
        summary["rgb_shape"] = list(env.render_rgb().shape)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
