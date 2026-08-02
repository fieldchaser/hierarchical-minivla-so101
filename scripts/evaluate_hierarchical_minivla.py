#!/usr/bin/env python3
"""Evaluate Hierarchical MiniVLA in closed-loop rendered MuJoCo rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hierarchical_minivla.flat_minivla import (
    encode_instructions,
    proprio_from_observation,
)
from hierarchical_minivla.hierarchical_minivla import load_hierarchical_minivla
from hierarchical_minivla.hierarchical_policy import (
    TRANSPORT_XY_TOLERANCE,
    MonotonicPhaseTracker,
    observed_event_next_phase,
    phase_progress_summary,
)
from hierarchical_minivla.scripted_expert import (
    PHASE_NAMES,
    instruction_for_goal,
    instruction_variant_for_episode,
    is_released_in_bin,
)

WORKSPACE_LOW = np.array([-0.35, 0.25, 0.045])
WORKSPACE_HIGH = np.array([0.05, 0.80, 0.25])


def furthest_milestone(
    success: bool,
    min_reach_distance: float,
    min_grasp_distance: float,
    first_close_step: int | None,
    first_lift_step: int | None,
    first_bin_aligned_step: int | None,
) -> str:
    if success:
        return "released"
    if first_bin_aligned_step is not None:
        return "bin_aligned"
    if first_lift_step is not None:
        return "lifted"
    if first_close_step is not None:
        return "close_command"
    if min_grasp_distance < 0.02:
        return "near_grasp"
    if min_reach_distance < 0.02:
        return "near_reach"
    return "initial"


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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1700)
    parser.add_argument("--cube-pos-std", type=float, default=0.006)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--phase-votes", type=int, default=3)
    parser.add_argument("--trace-every", type=int, default=25)
    parser.add_argument("--camera", default="front_close")
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.phase_votes < 1:
        parser.error("--phase-votes must be positive")
    if args.trace_every < 1:
        parser.error("--trace-every must be positive")

    hw3_path = args.eth_hw3.expanduser().resolve()
    xml_path = hw3_path / "so101_gym/assets/so100_multicube_ee.xml"
    if not xml_path.is_file():
        raise SystemExit(f"Multicube scene not found: {xml_path}")

    sys.path.insert(0, str(hw3_path))
    from hw3.sim_env import SO100MulticubeSimEnv

    device = resolve_device(args.device)
    policy, vocabulary = load_hierarchical_minivla(
        args.checkpoint.expanduser().resolve(), map_location=device
    )
    policy.eval()
    colors = ("red", "green", "blue")
    results = []
    for episode_index in range(args.episodes):
        seed = args.seed_start + episode_index
        goal_cube = colors[episode_index % len(colors)]
        instruction_variant = instruction_variant_for_episode(
            episode_index, len(colors)
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

        tracker = MonotonicPhaseTracker(required_votes=args.phase_votes)
        controller_transitions = [
            {"step": 0, "phase": PHASE_NAMES[tracker.current_phase]}
        ]
        observed_phase = 0
        observed_phase_steps = 0
        observed_transitions = [
            {"step": 0, "phase": PHASE_NAMES[observed_phase]}
        ]
        rollout_trace = []
        predicted_phase_counts: Counter[str] = Counter()
        predicted_observed_agreements = 0
        min_reach_distance = float("inf")
        min_grasp_distance = float("inf")
        max_cube_height = 0.0
        min_cube_bin_xy_distance = float("inf")
        closed_command_ever = False
        first_close_step = None
        first_open_after_close_step = None
        first_lift_step = None
        first_bin_aligned_step = None
        final_gripper_command = 1.0
        success = False
        for step in range(1, args.max_steps + 1):
            observation = env.get_obs()
            mocap = env.data.mocap_pos[env.mocap_id].copy()
            frame = env.render_rgb(args.camera)
            proprio = proprio_from_observation(observation, mocap)
            previous_controller_phase = tracker.current_phase
            delta, gripper, predicted_phase = policy.act(
                frame, tokens, proprio, phase=tracker.current_phase
            )
            predicted_phase_counts[PHASE_NAMES[predicted_phase]] += 1
            predicted_observed_agreements += int(predicted_phase == observed_phase)
            tracker.update(predicted_phase)
            controller_advanced = tracker.current_phase != previous_controller_phase
            if controller_advanced:
                controller_transitions.append(
                    {"step": step, "phase": PHASE_NAMES[tracker.current_phase]}
                )
                delta, gripper, _ = policy.act(
                    frame, tokens, proprio, phase=tracker.current_phase
                )

            final_gripper_command = gripper
            target = np.clip(mocap + delta, WORKSPACE_LOW, WORKSPACE_HIGH)
            env.set_mocap_pos(target)
            env.set_gripper(gripper)
            env.step()

            observed_phase_steps += 1
            observation = env.get_obs()
            mocap = env.data.mocap_pos[env.mocap_id].copy()
            next_observed_phase = observed_event_next_phase(
                observation,
                mocap,
                observed_phase,
                observed_phase_steps,
            )
            if next_observed_phase != observed_phase:
                observed_phase = next_observed_phase
                observed_transitions.append(
                    {"step": step, "phase": PHASE_NAMES[observed_phase]}
                )
                observed_phase_steps = 0

            cube_xyz = env.get_target_cube_state()[:3]
            bin_xyz = env.get_goal_pos()
            grasp_target = np.array(
                [cube_xyz[0] - 0.015, cube_xyz[1] - 0.004, 0.078]
            )
            reach_target = grasp_target.copy()
            reach_target[2] = 0.15
            reach_distance = float(np.linalg.norm(mocap - reach_target))
            grasp_distance = float(np.linalg.norm(mocap - grasp_target))
            min_reach_distance = min(
                min_reach_distance, reach_distance
            )
            min_grasp_distance = min(
                min_grasp_distance, grasp_distance
            )
            max_cube_height = max(max_cube_height, float(cube_xyz[2]))
            cube_bin_xy_distance = float(
                np.linalg.norm(cube_xyz[:2] - bin_xyz[:2])
            )
            min_cube_bin_xy_distance = min(
                min_cube_bin_xy_distance, cube_bin_xy_distance
            )
            closed_command_ever = closed_command_ever or gripper < 0.0
            if gripper < 0.0 and first_close_step is None:
                first_close_step = step
            if (
                first_close_step is not None
                and gripper > 0.0
                and first_open_after_close_step is None
            ):
                first_open_after_close_step = step
            if cube_xyz[2] > 0.08 and first_lift_step is None:
                first_lift_step = step
            if (
                cube_xyz[2] > 0.08
                and cube_bin_xy_distance < TRANSPORT_XY_TOLERANCE
                and first_bin_aligned_step is None
            ):
                first_bin_aligned_step = step
            success = is_released_in_bin(
                cube_xyz, bin_xyz, env.get_gripper_angle()
            )
            if step == 1 or step % args.trace_every == 0 or controller_advanced:
                rollout_trace.append(
                    {
                        "step": step,
                        "predicted_phase": PHASE_NAMES[predicted_phase],
                        "controller_phase": PHASE_NAMES[tracker.current_phase],
                        "observed_phase": PHASE_NAMES[observed_phase],
                        "action_delta": np.asarray(delta).round(6).tolist(),
                        "gripper_command": gripper,
                        "mocap_xyz": mocap.round(6).tolist(),
                        "reach_distance_m": reach_distance,
                        "grasp_distance_m": grasp_distance,
                        "cube_xyz": cube_xyz.round(6).tolist(),
                    }
                )
            if success:
                break

        env.renderer.close()
        milestone = furthest_milestone(
            success,
            min_reach_distance,
            min_grasp_distance,
            first_close_step,
            first_lift_step,
            first_bin_aligned_step,
        )
        result = {
            "seed": seed,
            "goal_cube": goal_cube,
            "instruction": instruction,
            "success": success,
            "steps": step,
            "controller_final_phase": PHASE_NAMES[tracker.current_phase],
            "controller_phase_transitions": controller_transitions,
            "observed_final_phase": PHASE_NAMES[observed_phase],
            "observed_phase_transitions": observed_transitions,
            "rollout_trace": rollout_trace,
            "predicted_phase_counts": {
                name: predicted_phase_counts.get(name, 0) for name in PHASE_NAMES
            },
            "predicted_observed_phase_agreement": (
                predicted_observed_agreements / step
            ),
            "furthest_milestone": milestone,
            "min_reach_distance_m": min_reach_distance,
            "min_grasp_distance_m": min_grasp_distance,
            "closed_command_ever": closed_command_ever,
            "first_close_step": first_close_step,
            "first_open_after_close_step": first_open_after_close_step,
            "first_lift_step": first_lift_step,
            "first_bin_aligned_step": first_bin_aligned_step,
            "final_gripper_command": final_gripper_command,
            "final_gripper_angle": env.get_gripper_angle(),
            "max_cube_height_m": max_cube_height,
            "min_cube_bin_xy_distance_m": min_cube_bin_xy_distance,
            "final_cube_xyz": env.get_target_cube_state()[:3].round(6).tolist(),
            "bin_xyz": env.get_goal_pos().round(6).tolist(),
            "final_mocap_xyz": env.data.mocap_pos[env.mocap_id].round(6).tolist(),
            "final_ee_xyz": env.get_ee_pos().round(6).tolist(),
        }
        results.append(result)
        print(
            f"[{episode_index + 1:02d}/{args.episodes:02d}] "
            f"seed={seed} goal={goal_cube} success={success} "
            f"controller={PHASE_NAMES[tracker.current_phase]} "
            f"observed={PHASE_NAMES[observed_phase]} "
            f"milestone={milestone} steps={step}"
        )

    num_successes = sum(result["success"] for result in results)
    controller_final_phases = [
        PHASE_NAMES.index(result["controller_final_phase"]) for result in results
    ]
    observed_final_phases = [
        PHASE_NAMES.index(result["observed_final_phase"]) for result in results
    ]
    summary = {
        "seed_start": args.seed_start,
        "num_successes": num_successes,
        "num_episodes": args.episodes,
        "success_rate": num_successes / args.episodes,
        "camera": args.camera,
        "device": str(device),
        "phase_votes": args.phase_votes,
        "num_closed_command_episodes": sum(
            result["closed_command_ever"] for result in results
        ),
        "num_open_after_close_episodes": sum(
            result["first_open_after_close_step"] is not None for result in results
        ),
        "num_lifted_episodes": sum(
            result["max_cube_height_m"] > 0.08 for result in results
        ),
        "num_bin_aligned_episodes": sum(
            result["min_cube_bin_xy_distance_m"] < TRANSPORT_XY_TOLERANCE
            for result in results
        ),
        "mean_min_grasp_distance_m": float(
            np.mean([result["min_grasp_distance_m"] for result in results])
        ),
        "mean_min_reach_distance_m": float(
            np.mean([result["min_reach_distance_m"] for result in results])
        ),
        "mean_predicted_observed_phase_agreement": float(
            np.mean(
                [result["predicted_observed_phase_agreement"] for result in results]
            )
        ),
        "furthest_milestone_counts": dict(
            Counter(result["furthest_milestone"] for result in results)
        ),
        "controller_phase_progress": phase_progress_summary(
            controller_final_phases
        ),
        "observed_phase_progress": phase_progress_summary(observed_final_phases),
        "results": results,
    }
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
