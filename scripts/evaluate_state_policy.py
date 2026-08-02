#!/usr/bin/env python3
"""Evaluate the learned state-and-goal policy in closed-loop MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from hierarchical_minivla.scripted_expert import is_released_in_bin
from hierarchical_minivla.state_policy import load_policy, observation_to_feature

WORKSPACE_LOW = np.array([-0.35, 0.25, 0.045])
WORKSPACE_HIGH = np.array([0.05, 0.80, 0.25])


class PhysicsOnlyRenderer:
    def __init__(self, *args, **kwargs) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-hw3", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--cube-pos-std", type=float, default=0.006)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")

    mujoco.Renderer = PhysicsOnlyRenderer
    sys.path.insert(0, str(hw3_path))
    from hw3.sim_env import SO100MulticubeSimEnv

    policy = load_policy(args.checkpoint.expanduser().resolve())
    policy.eval()
    colors = ("red", "green", "blue")
    results = []
    for episode_index in range(args.episodes):
        seed = args.seed_start + episode_index
        goal_cube = colors[episode_index % len(colors)]
        env = SO100MulticubeSimEnv(
            xml_path=xml_path,
            goal_cube=goal_cube,
            cube_pos_std=args.cube_pos_std,
            shuffle_cubes=True,
            seed=seed,
        )

        success = False
        for step in range(1, args.max_steps + 1):
            mocap = env.data.mocap_pos[env.mocap_id]
            mocap_feature = mocap if policy.feature_mean.numel() == 37 else None
            feature = observation_to_feature(env.get_obs(), mocap_feature)
            delta, gripper = policy.act(feature)
            target = np.clip(
                env.data.mocap_pos[env.mocap_id] + delta,
                WORKSPACE_LOW,
                WORKSPACE_HIGH,
            )
            env.set_mocap_pos(target)
            env.set_gripper(gripper)
            env.step()
            success = is_released_in_bin(
                env.get_target_cube_state()[:3],
                env.get_goal_pos(),
                env.get_gripper_angle(),
            )
            if success:
                break

        final_cube_xyz = env.get_target_cube_state()[:3]
        result = {
            "seed": seed,
            "goal_cube": goal_cube,
            "success": success,
            "steps": step,
            "final_cube_xyz": final_cube_xyz.round(6).tolist(),
            "bin_xyz": env.get_goal_pos().round(6).tolist(),
        }
        results.append(result)
        print(
            f"[{episode_index + 1:02d}/{args.episodes:02d}] "
            f"seed={seed} goal={goal_cube} success={success} steps={step}"
        )

    num_successes = sum(result["success"] for result in results)
    summary = {
        "seed_start": args.seed_start,
        "num_successes": num_successes,
        "num_episodes": args.episodes,
        "success_rate": num_successes / args.episodes,
        "results": results,
    }
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
