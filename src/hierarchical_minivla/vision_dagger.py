"""Visual DAgger collection with expert labels on learner-visited states."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .dagger import ScriptedRecoveryOracle, WORKSPACE_HIGH, WORKSPACE_LOW
from .flat_minivla import proprio_from_observation
from .hierarchical_minivla import HierarchicalMiniVLA
from .hierarchical_policy import MonotonicPhaseTracker
from .scripted_expert import PHASE_NAMES, ScriptedEpisode, is_released_in_bin


def run_visual_dagger_episode(
    env: Any,
    policy: HierarchicalMiniVLA,
    tokens: np.ndarray,
    instruction: str,
    frame_observer: Callable[[], np.ndarray],
    learner_reach_steps: int = 120,
    max_steps: int = 800,
    transition_votes: int = 3,
) -> ScriptedEpisode:
    """Execute the learner during reach and label every state with the expert."""
    if learner_reach_steps < 0:
        raise ValueError("learner_reach_steps must be non-negative")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if transition_votes < 1:
        raise ValueError("transition_votes must be positive")

    records: dict[str, list[np.ndarray | int | float | bool | str]] = {
        "rgb": [],
        "instruction": [],
        "state_ee_xyz": [],
        "state_mocap_xyz": [],
        "state_joints": [],
        "state_gripper": [],
        "cubes_xyz": [],
        "cubes": [],
        "state_goal": [],
        "goal_pos": [],
        "action_ee_xyz": [],
        "action_gripper": [],
        "executed_action_ee_xyz": [],
        "executed_action_gripper": [],
        "phase": [],
        "learner_phase": [],
        "learner_predicted_phase": [],
        "dagger": [],
        "recovery": [],
    }
    oracle = ScriptedRecoveryOracle()
    learner_tracker = MonotonicPhaseTracker(required_votes=transition_votes)
    learner_steps_used = 0

    for _ in range(max_steps):
        oracle.maybe_advance(env)
        observation = env.get_obs()
        mocap = env.data.mocap_pos[env.mocap_id].copy()
        frame = np.asarray(frame_observer())
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("RGB frames must be uint8 arrays with shape H x W x 3")
        proprio = proprio_from_observation(observation, mocap)
        previous_phase = learner_tracker.current_phase
        learner_delta, learner_gripper, predicted_phase = policy.act(
            frame,
            tokens,
            proprio,
            phase=learner_tracker.current_phase,
        )
        learner_tracker.update(predicted_phase)
        if learner_tracker.current_phase != previous_phase:
            learner_delta, learner_gripper, _ = policy.act(
                frame,
                tokens,
                proprio,
                phase=learner_tracker.current_phase,
            )

        expert_delta, expert_gripper, expert_phase = oracle.label(env)
        learner_executed = (
            oracle.phase == 0 and learner_steps_used < learner_reach_steps
        )
        if learner_executed:
            executed_delta = learner_delta
            executed_gripper = learner_gripper
            learner_steps_used += 1
        else:
            executed_delta = expert_delta
            executed_gripper = expert_gripper

        records["rgb"].append(frame.copy())
        records["instruction"].append(instruction)
        records["state_ee_xyz"].append(observation["ee_pos"].copy())
        records["state_mocap_xyz"].append(mocap)
        records["state_joints"].append(observation["joints"].copy())
        records["state_gripper"].append(observation["gripper"].copy())
        records["cubes_xyz"].append(observation["cubes_xyz"].copy())
        records["cubes"].append(observation["cubes"].copy())
        records["state_goal"].append(observation["goal"].copy())
        records["goal_pos"].append(observation["goal_pos"].copy())
        records["action_ee_xyz"].append(expert_delta)
        records["action_gripper"].append(expert_gripper)
        records["executed_action_ee_xyz"].append(executed_delta)
        records["executed_action_gripper"].append(executed_gripper)
        records["phase"].append(expert_phase)
        records["learner_phase"].append(learner_tracker.current_phase)
        records["learner_predicted_phase"].append(predicted_phase)
        records["dagger"].append(learner_executed)
        records["recovery"].append(not learner_executed)

        target = np.clip(mocap + executed_delta, WORKSPACE_LOW, WORKSPACE_HIGH)
        env.set_mocap_pos(target)
        env.set_gripper(executed_gripper)
        env.step()
        oracle.note_step(not learner_executed)

        if is_released_in_bin(
            env.get_target_cube_state()[:3],
            env.get_goal_pos(),
            env.get_gripper_angle(),
        ):
            break

    final_cube_xyz = env.get_target_cube_state()[:3].copy()
    bin_xyz = env.get_goal_pos().copy()
    success = is_released_in_bin(
        final_cube_xyz,
        bin_xyz,
        env.get_gripper_angle(),
    )
    integer_keys = {"phase", "learner_phase", "learner_predicted_phase"}
    boolean_keys = {"dagger", "recovery"}
    arrays = {}
    for key, values in records.items():
        dtype = np.float32
        if key in integer_keys:
            dtype = np.int64
        elif key in boolean_keys:
            dtype = np.bool_
        elif key == "rgb":
            dtype = np.uint8
        elif key == "instruction":
            dtype = np.str_
        arrays[key] = np.asarray(values, dtype=dtype)
    arrays["action_gripper"] = arrays["action_gripper"].reshape(-1, 1)
    arrays["executed_action_gripper"] = arrays[
        "executed_action_gripper"
    ].reshape(-1, 1)
    arrays["phase_names"] = np.asarray(PHASE_NAMES)
    return ScriptedEpisode(
        success=success,
        arrays=arrays,
        final_cube_xyz=final_cube_xyz,
        bin_xyz=bin_xyz,
        grasp_contact_angle=oracle.grasp_contact_angle,
    )
