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
    INSTRUCTION_TEMPLATES,
    instruction_for_goal,
    run_scripted_episode,
    save_scripted_episode,
)
from hierarchical_minivla.vision_data import validate_vision_episode_arrays


class PhysicsOnlyRenderer:
    """Avoid creating an OpenGL context during headless data collection."""

    def __init__(self, *args, **kwargs) -> None:
        pass


def save_rgb_preview(frames: np.ndarray, path: Path, num_frames: int = 6) -> Path:
    """Save evenly spaced RGB frames as a horizontal JPEG contact sheet."""
    import cv2

    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    contact_sheet = np.concatenate(frames[indices], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(
        str(path), cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR)
    )
    if not written:
        raise RuntimeError(f"Failed to save RGB preview: {path}")
    return path


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
    parser.add_argument("--record-rgb", action="store_true")
    parser.add_argument("--camera", default="front_close")
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--preview-path", type=Path)
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
    if args.render_width <= 0 or args.render_height <= 0:
        parser.error("render dimensions must be positive")
    if args.preview_path is not None and not args.record_rgb:
        parser.error("--preview-path requires --record-rgb")

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.record_rgb:
        mujoco.Renderer = PhysicsOnlyRenderer
    sys.path.insert(0, str(hw3_path))
    from hw3.sim_env import SO100MulticubeSimEnv

    attempts = []
    num_successes = 0
    preview_path = None
    for episode_index in range(args.episodes):
        seed = args.seed_start + episode_index
        goal_cube = args.colors[episode_index % len(args.colors)]
        env = SO100MulticubeSimEnv(
            xml_path=xml_path,
            goal_cube=goal_cube,
            cube_pos_std=args.cube_pos_std,
            shuffle_cubes=True,
            seed=seed,
            render_w=args.render_width,
            render_h=args.render_height,
        )
        recovery_rng = np.random.default_rng(seed + 1_000_000)
        instruction = instruction_for_goal(
            goal_cube, variant=seed % len(INSTRUCTION_TEMPLATES)
        )
        frame_observer = None
        if args.record_rgb:
            frame_observer = lambda: env.render_rgb(args.camera)
        episode = run_scripted_episode(
            env,
            recovery_rng=recovery_rng,
            recovery_pos_std=args.recovery_pos_std,
            frame_observer=frame_observer,
            instruction=instruction,
        )
        vision_summary = None
        if args.record_rgb:
            vision_summary = validate_vision_episode_arrays(episode.arrays)

        trajectory = None
        recovery_steps = int(episode.arrays["recovery"].sum())
        if episode.success:
            file_name = f"episode_{episode_index:04d}_seed_{seed:04d}_{goal_cube}.npz"
            save_scripted_episode(episode, output_dir / file_name)
            trajectory = file_name
            num_successes += 1
            if args.preview_path is not None and preview_path is None:
                preview_path = save_rgb_preview(
                    episode.arrays["rgb"], args.preview_path.expanduser().resolve()
                )

        record = {
            "episode_index": episode_index,
            "seed": seed,
            "goal_cube": goal_cube,
            "instruction": instruction if args.record_rgb else None,
            "vision": vision_summary,
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
        if args.record_rgb:
            env.renderer.close()

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
            "record_rgb": args.record_rgb,
            "camera": args.camera if args.record_rgb else None,
            "render_width": args.render_width if args.record_rgb else None,
            "render_height": args.render_height if args.record_rgb else None,
            "instruction_templates": (
                list(INSTRUCTION_TEMPLATES) if args.record_rgb else None
            ),
        },
        "num_successes": num_successes,
        "success_rate": success_rate,
        "num_saved_steps": sum(
            record["num_steps"] for record in attempts if record["success"]
        ),
        "num_saved_recovery_steps": sum(
            record["recovery_steps"] for record in attempts if record["success"]
        ),
        "preview": str(preview_path) if preview_path is not None else None,
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
                "preview": manifest["preview"],
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
