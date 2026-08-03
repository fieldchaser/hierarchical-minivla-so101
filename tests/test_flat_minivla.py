from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from hierarchical_minivla.flat_minivla import (
    FlatMiniVLA,
    build_vocabulary,
    encode_instructions,
    load_vision_episodes,
    load_flat_minivla,
    phase_balanced_indices,
    proprio_from_observation,
    save_flat_minivla,
    split_vision_episode_paths,
    transition_focused_mask,
)


class FlatMiniVLATest(unittest.TestCase):
    def test_vocabulary_and_encoding_are_deterministic(self) -> None:
        instructions = ["Pick up the red cube.", "Pick up the blue cube."]
        vocabulary = build_vocabulary(instructions)
        encoded = encode_instructions(instructions, vocabulary)
        self.assertEqual(vocabulary["<pad>"], 0)
        self.assertEqual(encoded.shape, (2, 5))
        self.assertNotEqual(vocabulary["red"], vocabulary["blue"])

    def test_phase_balanced_indices_cover_all_phases(self) -> None:
        phases = np.repeat(np.arange(6), 10)
        selected = phase_balanced_indices(phases, batch_size=12)
        np.testing.assert_array_equal(np.bincount(phases[selected]), np.full(6, 2))

    def test_policy_forward_and_backward(self) -> None:
        policy = FlatMiniVLA(
            vocab_size=8,
            proprio_mean=np.zeros(13),
            proprio_std=np.ones(13),
            delta_mean=np.zeros(3),
            delta_std=np.ones(3),
        )
        rgb = torch.zeros((2, 32, 32, 3), dtype=torch.uint8)
        tokens = torch.tensor([[2, 3, 0], [2, 4, 5]])
        proprio = torch.zeros((2, 13))
        delta = torch.zeros((2, 3))
        gripper = torch.tensor([1.0, 0.0])
        loss, _, _ = policy.loss(rgb, tokens, proprio, delta, gripper)
        loss.backward()
        self.assertEqual(policy(rgb, tokens, proprio)[0].shape, (2, 3))
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))

    def test_proprioception_excludes_scene_truth(self) -> None:
        observation = {
            "ee_pos": np.array([1.0, 2.0, 3.0]),
            "joints": np.arange(6.0),
            "gripper": np.array([0.5]),
            "cubes_xyz": np.full(9, 99.0),
            "goal_pos": np.full(3, 99.0),
        }
        command_xyz = np.array([7.0, 8.0, 9.0])
        proprio = proprio_from_observation(observation, command_xyz)
        np.testing.assert_array_equal(
            proprio,
            np.concatenate(
                [observation["ee_pos"], command_xyz, np.arange(6.0), [0.5]]
            ),
        )

    def test_act_clips_delta_and_decodes_gripper(self) -> None:
        policy = FlatMiniVLA(
            vocab_size=4,
            proprio_mean=np.zeros(13),
            proprio_std=np.ones(13),
            delta_mean=np.zeros(3),
            delta_std=np.ones(3),
        )
        for parameter in policy.parameters():
            parameter.data.zero_()
        policy.action_head[-1].bias.data.copy_(
            torch.tensor([10.0, -10.0, 0.0, -10.0])
        )
        delta, gripper = policy.act(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.array([2, 3]),
            np.zeros(13, dtype=np.float32),
        )
        np.testing.assert_allclose(delta, [0.004, -0.004, 0.0])
        self.assertEqual(gripper, -0.174)

    def test_checkpoint_round_trip(self) -> None:
        policy = FlatMiniVLA(
            vocab_size=4,
            proprio_mean=np.zeros(13),
            proprio_std=np.ones(13),
            delta_mean=np.zeros(3),
            delta_std=np.ones(3),
        )
        vocabulary = {"<pad>": 0, "<unk>": 1, "pick": 2, "red": 3}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_flat_minivla(policy, vocabulary, path)
            restored, restored_vocabulary = load_flat_minivla(path)
        self.assertEqual(restored_vocabulary, vocabulary)
        for expected, actual in zip(policy.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_episode_split_holds_out_each_instruction_without_overlap(self) -> None:
        with TemporaryDirectory() as directory:
            paths = []
            for instruction_index in range(3):
                for episode_index in range(3):
                    path = Path(directory) / f"episode_{instruction_index}_{episode_index}.npz"
                    np.savez_compressed(
                        path,
                        instruction=np.repeat(
                            f"instruction {instruction_index}", repeats=2
                        ),
                    )
                    paths.append(path)
            train, validation = split_vision_episode_paths(paths, seed=4)
        self.assertEqual(len(train), 6)
        self.assertEqual(len(validation), 3)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertEqual(
            {path.name.split("_")[1] for path in validation}, {"0", "1", "2"}
        )

    def test_transition_filter_removes_ambiguous_waits(self) -> None:
        proprio = np.zeros((5, 13), dtype=np.float32)
        proprio[:, -1] = [1.0, 1.0, 0.8, 0.32, 1.0]
        arrays = {
            "phase": np.array([0, 0, 2, 2, 5]),
            "delta": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.001, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "proprio": proprio,
        }
        np.testing.assert_array_equal(
            transition_focused_mask(arrays),
            [False, True, True, False, True],
        )

    def test_episode_loader_can_select_only_dagger_frames(self) -> None:
        num_steps = 4
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            np.savez_compressed(
                path,
                rgb=np.zeros((num_steps, 8, 8, 3), dtype=np.uint8),
                instruction=np.asarray(["Pick up the red cube."] * num_steps),
                state_ee_xyz=np.zeros((num_steps, 3), dtype=np.float32),
                state_mocap_xyz=np.zeros((num_steps, 3), dtype=np.float32),
                state_joints=np.zeros((num_steps, 6), dtype=np.float32),
                state_gripper=np.ones((num_steps, 1), dtype=np.float32),
                cubes_xyz=np.zeros((num_steps, 9), dtype=np.float32),
                state_goal=np.zeros((num_steps, 3), dtype=np.float32),
                goal_pos=np.zeros((num_steps, 3), dtype=np.float32),
                action_ee_xyz=np.arange(num_steps * 3, dtype=np.float32).reshape(
                    num_steps, 3
                ),
                action_gripper=np.ones((num_steps, 1), dtype=np.float32),
                phase=np.zeros(num_steps, dtype=np.int64),
                dagger=np.asarray([True, False, True, False]),
                success=np.asarray(True),
            )
            arrays = load_vision_episodes([path], frame_mask_key="dagger")

        self.assertEqual(len(arrays["rgb"]), 2)
        np.testing.assert_array_equal(
            arrays["delta"], [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]
        )


if __name__ == "__main__":
    unittest.main()
