#!/usr/bin/env python3
"""Collect one round of on-policy hierarchical DAgger data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco

from hierarchical_minivla.dagger import run_dagger_episode
from hierarchical_minivla.hierarchical_policy import load_hierarchical_policy
from hierarchical_minivla.scripted_expert import PHASE_NAMES, save_scripted_episode


class PhysicsOnlyRenderer:
    def __init__(self, *args, **kwargs) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-hw3", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/dagger/round_1")
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=300)
    parser.add_argument("--cube-pos-std", type=float, default=0.006)
    parser.add_argument("--learner-steps-per-phase", type=int, default=80)
    parser.add_argument("--learner-grasp-lift-steps", type=int, default=0)
    parser.add_argument("--observed-phase-events", action="store_true")
    parser.add_argument("--learner-stagnation-steps", type=int, default=5)
    parser.add_argument("--learner-action-threshold", type=float, default=0.0002)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    args = parser.parse_args()

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mujoco.Renderer = PhysicsOnlyRenderer
    sys.path.insert(0, str(hw3_path))
    from hw3.sim_env import SO100MulticubeSimEnv

    policy = load_hierarchical_policy(args.checkpoint.expanduser().resolve()).eval()
    if policy.feature_mean.numel() != 37:
        raise SystemExit("DAgger collection requires a 37-dimensional mocap checkpoint")

    colors = ("red", "green", "blue")
    attempts = []
    num_successes = 0
    saved_steps = 0
    learner_state_steps = 0
    learner_phase_counts = {name: 0 for name in PHASE_NAMES}
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
        episode = run_dagger_episode(
            env,
            policy,
            learner_steps_per_phase=args.learner_steps_per_phase,
            learner_grasp_lift_steps=args.learner_grasp_lift_steps,
            use_observed_phase_events=args.observed_phase_events,
            learner_stagnation_steps=args.learner_stagnation_steps,
            learner_action_threshold=args.learner_action_threshold,
            max_steps=args.max_steps,
        )
        dagger_steps = int(episode.arrays["dagger"].sum())
        trajectory = None
        if episode.success:
            file_name = f"episode_{episode_index:04d}_seed_{seed:04d}_{goal_cube}.npz"
            save_scripted_episode(episode, output_dir / file_name)
            trajectory = file_name
            num_successes += 1
            saved_steps += episode.num_steps
            learner_state_steps += dagger_steps
            for phase in episode.arrays["phase"][episode.arrays["dagger"]]:
                learner_phase_counts[PHASE_NAMES[int(phase)]] += 1
        record = {
            "episode_index": episode_index,
            "seed": seed,
            "goal_cube": goal_cube,
            "success": episode.success,
            "num_steps": episode.num_steps,
            "learner_state_steps": dagger_steps,
            "final_phase": int(episode.arrays["phase"][-1]),
            "final_cube_xyz": episode.final_cube_xyz.round(6).tolist(),
            "bin_xyz": episode.bin_xyz.round(6).tolist(),
            "trajectory": trajectory,
        }
        attempts.append(record)
        print(
            f"[{episode_index + 1:02d}/{args.episodes:02d}] seed={seed} "
            f"goal={goal_cube} success={episode.success} "
            f"steps={episode.num_steps} learner_states={dagger_steps}"
        )

    success_rate = num_successes / args.episodes
    manifest = {
        "settings": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "cube_pos_std": args.cube_pos_std,
            "learner_steps_per_phase": args.learner_steps_per_phase,
            "learner_grasp_lift_steps": args.learner_grasp_lift_steps,
            "observed_phase_events": args.observed_phase_events,
            "learner_stagnation_steps": args.learner_stagnation_steps,
            "learner_action_threshold": args.learner_action_threshold,
            "max_steps": args.max_steps,
            "checkpoint": str(args.checkpoint),
        },
        "num_successes": num_successes,
        "success_rate": success_rate,
        "num_saved_steps": saved_steps,
        "num_learner_state_steps": learner_state_steps,
        "learner_phase_counts": learner_phase_counts,
        "attempts": attempts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {key: value for key, value in manifest.items() if key != "attempts"}
    print(json.dumps(summary, indent=2))
    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"Success rate {success_rate:.1%} is below {args.min_success_rate:.1%}"
        )


if __name__ == "__main__":
    main()
