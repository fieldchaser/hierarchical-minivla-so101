#!/usr/bin/env python3
"""Collect RGB DAgger corrections from Hierarchical MiniVLA rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hierarchical_minivla.flat_minivla import encode_instructions
from hierarchical_minivla.hierarchical_minivla import load_hierarchical_minivla
from hierarchical_minivla.scripted_expert import (
    PHASE_NAMES,
    instruction_for_goal,
    instruction_variant_for_episode,
    save_scripted_episode,
)
from hierarchical_minivla.vision_dagger import run_visual_dagger_episode
from hierarchical_minivla.vision_data import validate_vision_episode_arrays


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-hw3", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/vision/dagger_round_1")
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1609)
    parser.add_argument("--cube-pos-std", type=float, default=0.006)
    parser.add_argument("--learner-reach-steps", type=int, default=120)
    parser.add_argument("--phase-votes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--camera", default="front_close")
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument(
        "--colors",
        nargs="+",
        choices=("red", "green", "blue"),
        default=("red", "green", "blue"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.learner_reach_steps < 0:
        parser.error("--learner-reach-steps must be non-negative")
    if args.phase_votes < 1:
        parser.error("--phase-votes must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be between 0 and 1")

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(hw3_path))
    from hw3.sim_env import SO100MulticubeSimEnv

    device = resolve_device(args.device)
    policy, vocabulary = load_hierarchical_minivla(
        args.checkpoint.expanduser().resolve(), map_location=device
    )
    policy.eval()
    attempts = []
    num_successes = 0
    num_saved_steps = 0
    num_learner_steps = 0
    expert_phase_counts = {name: 0 for name in PHASE_NAMES}
    predicted_phase_counts = {name: 0 for name in PHASE_NAMES}
    for episode_index in range(args.episodes):
        seed = args.seed_start + episode_index
        goal_cube = args.colors[episode_index % len(args.colors)]
        instruction_variant = instruction_variant_for_episode(
            episode_index, len(args.colors)
        )
        instruction = instruction_for_goal(goal_cube, instruction_variant)
        tokens = encode_instructions([instruction], vocabulary)[0]
        env = SO100MulticubeSimEnv(
            xml_path=xml_path,
            goal_cube=goal_cube,
            cube_pos_std=args.cube_pos_std,
            shuffle_cubes=True,
            seed=seed,
            render_w=args.render_width,
            render_h=args.render_height,
        )
        episode = run_visual_dagger_episode(
            env,
            policy,
            tokens,
            instruction,
            frame_observer=lambda: env.render_rgb(args.camera),
            learner_reach_steps=args.learner_reach_steps,
            max_steps=args.max_steps,
            transition_votes=args.phase_votes,
        )
        vision_summary = validate_vision_episode_arrays(episode.arrays)
        dagger_mask = np.asarray(episode.arrays["dagger"], dtype=bool)
        learner_steps = int(dagger_mask.sum())
        trajectory = None
        if episode.success:
            file_name = f"episode_{episode_index:04d}_seed_{seed:04d}_{goal_cube}.npz"
            save_scripted_episode(episode, output_dir / file_name)
            trajectory = file_name
            num_successes += 1
            num_saved_steps += episode.num_steps
            num_learner_steps += learner_steps
            for phase in episode.arrays["phase"][dagger_mask]:
                expert_phase_counts[PHASE_NAMES[int(phase)]] += 1
            for phase in episode.arrays["learner_predicted_phase"][dagger_mask]:
                predicted_phase_counts[PHASE_NAMES[int(phase)]] += 1
        attempts.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "goal_cube": goal_cube,
                "instruction": instruction,
                "success": episode.success,
                "num_steps": episode.num_steps,
                "learner_steps": learner_steps,
                "vision": vision_summary,
                "trajectory": trajectory,
            }
        )
        env.renderer.close()
        print(
            f"[{episode_index + 1:02d}/{args.episodes:02d}] seed={seed} "
            f"goal={goal_cube} success={episode.success} "
            f"steps={episode.num_steps} learner_states={learner_steps}"
        )

    success_rate = num_successes / args.episodes
    manifest = {
        "settings": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "cube_pos_std": args.cube_pos_std,
            "learner_reach_steps": args.learner_reach_steps,
            "phase_votes": args.phase_votes,
            "max_steps": args.max_steps,
            "camera": args.camera,
            "render_width": args.render_width,
            "render_height": args.render_height,
            "colors": list(args.colors),
            "checkpoint": str(args.checkpoint),
        },
        "num_successes": num_successes,
        "success_rate": success_rate,
        "num_saved_steps": num_saved_steps,
        "num_learner_state_steps": num_learner_steps,
        "learner_expert_phase_counts": expert_phase_counts,
        "learner_predicted_phase_counts": predicted_phase_counts,
        "attempts": attempts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {key: value for key, value in manifest.items() if key != "attempts"}
    print(json.dumps(summary, indent=2))
    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"Success rate {success_rate:.1%} is below "
            f"{args.min_success_rate:.1%}"
        )


if __name__ == "__main__":
    main()
