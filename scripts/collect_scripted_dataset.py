#!/usr/bin/env python3
"""Collect a randomized, goal-conditioned MuJoCo demonstration dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from hierarchical_minivla.scripted_expert import (
    run_scripted_episode,
    save_scripted_episode,
)


class PhysicsOnlyRenderer:
    """Avoid creating an OpenGL context during headless data collection."""

    def __init__(self, *args, **kwargs) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-hw3", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scripted/randomized"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--cube-pos-std", type=float, default=0.006)
    parser.add_argument("--recovery-pos-std", type=float, default=0.0)
    parser.add_argument(
        "--colors",
        nargs="+",
        choices=("red", "green", "blue"),
        default=("red", "green", "blue"),
    )
    parser.add_argument("--min-success-rate", type=float, default=0.9)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be between 0 and 1")
    if args.recovery_pos_std < 0:
        parser.error("--recovery-pos-std must be non-negative")

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mujoco.Renderer = PhysicsOnlyRenderer
    sys.path.insert(0, str(hw3_path))
    from hw3.sim_env import SO100MulticubeSimEnv

    attempts = []
    num_successes = 0
    for episode_index in range(args.episodes):
        seed = args.seed_start + episode_index
        goal_cube = args.colors[episode_index % len(args.colors)]
        env = SO100MulticubeSimEnv(
            xml_path=xml_path,
            goal_cube=goal_cube,
            cube_pos_std=args.cube_pos_std,
            shuffle_cubes=True,
            seed=seed,
        )
        recovery_rng = np.random.default_rng(seed + 1_000_000)
        episode = run_scripted_episode(
            env,
            recovery_rng=recovery_rng,
            recovery_pos_std=args.recovery_pos_std,
        )

        trajectory = None
        recovery_steps = int(episode.arrays["recovery"].sum())
        if episode.success:
            file_name = f"episode_{episode_index:04d}_seed_{seed:04d}_{goal_cube}.npz"
            save_scripted_episode(episode, output_dir / file_name)
            trajectory = file_name
            num_successes += 1

        record = {
            "episode_index": episode_index,
            "seed": seed,
            "goal_cube": goal_cube,
            "success": episode.success,
            "num_steps": episode.num_steps,
            "recovery_steps": recovery_steps,
            "grasp_contact_angle": round(episode.grasp_contact_angle, 6),
            "final_cube_xyz": episode.final_cube_xyz.round(6).tolist(),
            "bin_xyz": episode.bin_xyz.round(6).tolist(),
            "trajectory": trajectory,
        }
        attempts.append(record)
        status = "saved" if episode.success else "failed"
        print(
            f"[{episode_index + 1:02d}/{args.episodes:02d}] "
            f"seed={seed} goal={goal_cube} {status} steps={episode.num_steps}"
        )

    success_rate = num_successes / args.episodes
    manifest = {
        "settings": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "cube_pos_std": args.cube_pos_std,
            "recovery_pos_std": args.recovery_pos_std,
            "recovery_seed_offset": 1_000_000,
            "colors": list(args.colors),
            "shuffle_cubes": True,
        },
        "num_successes": num_successes,
        "success_rate": success_rate,
        "num_saved_steps": sum(
            record["num_steps"] for record in attempts if record["success"]
        ),
        "num_saved_recovery_steps": sum(
            record["recovery_steps"] for record in attempts if record["success"]
        ),
        "attempts": attempts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "num_successes": num_successes,
                "num_attempts": args.episodes,
                "success_rate": success_rate,
                "num_saved_steps": manifest["num_saved_steps"],
                "num_saved_recovery_steps": manifest["num_saved_recovery_steps"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"Success rate {success_rate:.1%} is below the required "
            f"{args.min_success_rate:.1%}"
        )


if __name__ == "__main__":
    main()
