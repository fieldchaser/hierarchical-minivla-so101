"""On-policy DAgger collection with scripted expert recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hierarchical_policy import HierarchicalStateGoalPolicy, MonotonicPhaseTracker
from .scripted_expert import (
    PHASE_NAMES,
    ScriptedEpisode,
    bounded_delta,
    is_released_in_bin,
)
from .state_policy import GRIPPER_CLOSED, GRIPPER_OPEN, observation_to_feature

WORKSPACE_LOW = np.array([-0.35, 0.25, 0.045])
WORKSPACE_HIGH = np.array([0.05, 0.80, 0.25])


@dataclass
class ScriptedRecoveryOracle:
    """Label arbitrary learner states and advance through physical phase gates."""

    phase: int = 0
    expert_steps_in_phase: int = 0
    lift_target: np.ndarray | None = None
    release_target: np.ndarray | None = None
    grasp_contact_angle: float = float("nan")

    def _grasp_target(self, env: Any) -> np.ndarray:
        cube_xyz = env.get_target_cube_state()[:3]
        return np.array([cube_xyz[0] - 0.015, cube_xyz[1] - 0.004, 0.078])

    def target(self, env: Any) -> np.ndarray:
        mocap = env.data.mocap_pos[env.mocap_id].copy()
        if self.phase == 0:
            target = self._grasp_target(env)
            target[2] = 0.15
            return target
        if self.phase == 1:
            return self._grasp_target(env)
        if self.phase == 3:
            assert self.lift_target is not None
            return self.lift_target
        if self.phase == 4:
            cube_xyz = env.get_target_cube_state()[:3]
            bin_xyz = env.get_goal_pos()
            held_offset = mocap - cube_xyz
            return np.array(
                [bin_xyz[0] + held_offset[0], bin_xyz[1] + held_offset[1], 0.17]
            )
        if self.phase == 5:
            assert self.release_target is not None
            return self.release_target
        return mocap

    def label(self, env: Any) -> tuple[np.ndarray, float, int]:
        mocap = env.data.mocap_pos[env.mocap_id]
        gripper = GRIPPER_OPEN if self.phase in (0, 1, 5) else GRIPPER_CLOSED
        return bounded_delta(mocap, self.target(env)), gripper, self.phase

    def maybe_advance(self, env: Any) -> bool:
        mocap = env.data.mocap_pos[env.mocap_id]
        cube_xyz = env.get_target_cube_state()[:3]
        bin_xyz = env.get_goal_pos()
        next_phase = self.phase

        if self.phase == 0 and np.linalg.norm(mocap - self.target(env)) < 0.004:
            next_phase = 1
        elif self.phase == 1 and np.linalg.norm(mocap - self.target(env)) < 0.004:
            next_phase = 2
        elif (
            self.phase == 2
            and self.expert_steps_in_phase >= 30
            and env.get_gripper_angle() < 0.6
        ):
            self.grasp_contact_angle = env.get_gripper_angle()
            self.lift_target = mocap.copy()
            self.lift_target[2] = 0.17
            next_phase = 3
        elif (
            self.phase == 3
            and cube_xyz[2] > 0.08
            and np.linalg.norm(mocap - self.target(env)) < 0.008
        ):
            next_phase = 4
        elif (
            self.phase == 4
            and cube_xyz[2] > 0.08
            and np.linalg.norm(cube_xyz[:2] - bin_xyz[:2]) < 0.012
        ):
            self.release_target = mocap.copy()
            self.release_target[2] = min(0.22, self.release_target[2] + 0.05)
            next_phase = 5

        if next_phase == self.phase:
            return False
        self.phase = next_phase
        self.expert_steps_in_phase = 0
        return True

    def note_step(self, expert_executed: bool) -> None:
        if expert_executed:
            self.expert_steps_in_phase += 1


def run_dagger_episode(
    env: Any,
    policy: HierarchicalStateGoalPolicy,
    learner_steps_per_phase: int = 80,
    learner_stagnation_steps: int = 5,
    learner_action_threshold: float = 0.0002,
    max_steps: int = 800,
    transition_votes: int = 3,
) -> ScriptedEpisode:
    """Collect expert labels on states produced by alternating learner/expert control."""
    if learner_steps_per_phase < 0:
        raise ValueError("learner_steps_per_phase must be non-negative")

    records: dict[str, list[np.ndarray | int | float | bool]] = {
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
    learner_stagnation_count = 0

    for _ in range(max_steps):
        if oracle.maybe_advance(env):
            learner_steps_used = 0
            learner_stagnation_count = 0

        obs = env.get_obs()
        mocap = env.data.mocap_pos[env.mocap_id].copy()
        feature = observation_to_feature(obs, mocap)
        predicted_phase = policy.predict_phase(feature)
        learner_phase = learner_tracker.update(predicted_phase)
        expert_delta, expert_gripper, expert_phase = oracle.label(env)

        learner_budget = learner_steps_per_phase if oracle.phase in (0, 1) else 0
        learner_executed = (
            learner_steps_used < learner_budget
            and learner_stagnation_count < learner_stagnation_steps
        )
        if learner_executed:
            executed_delta, executed_gripper, _ = policy.act(
                feature, phase=learner_phase
            )
            learner_steps_used += 1
            if np.linalg.norm(executed_delta) < learner_action_threshold:
                learner_stagnation_count += 1
            else:
                learner_stagnation_count = 0
        else:
            executed_delta = expert_delta
            executed_gripper = expert_gripper

        records["state_ee_xyz"].append(obs["ee_pos"].copy())
        records["state_mocap_xyz"].append(mocap)
        records["state_joints"].append(obs["joints"].copy())
        records["state_gripper"].append(obs["gripper"].copy())
        records["cubes_xyz"].append(obs["cubes_xyz"].copy())
        records["cubes"].append(obs["cubes"].copy())
        records["state_goal"].append(obs["goal"].copy())
        records["goal_pos"].append(obs["goal_pos"].copy())
        records["action_ee_xyz"].append(expert_delta)
        records["action_gripper"].append(expert_gripper)
        records["executed_action_ee_xyz"].append(executed_delta)
        records["executed_action_gripper"].append(executed_gripper)
        records["phase"].append(expert_phase)
        records["learner_phase"].append(learner_phase)
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
        arrays[key] = np.asarray(values, dtype=dtype)
    arrays["action_gripper"] = arrays["action_gripper"].reshape(-1, 1)
    arrays["executed_action_gripper"] = arrays["executed_action_gripper"].reshape(-1, 1)
    arrays["phase_names"] = np.asarray(PHASE_NAMES)
    return ScriptedEpisode(
        success=success,
        arrays=arrays,
        final_cube_xyz=final_cube_xyz,
        bin_xyz=bin_xyz,
        grasp_contact_angle=oracle.grasp_contact_angle,
    )
