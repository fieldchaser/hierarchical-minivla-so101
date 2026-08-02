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
    load_flat_minivla,
    phase_balanced_indices,
    save_flat_minivla,
    split_vision_episode_paths,
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
            proprio_mean=np.zeros(10),
            proprio_std=np.ones(10),
            delta_mean=np.zeros(3),
            delta_std=np.ones(3),
        )
        rgb = torch.zeros((2, 32, 32, 3), dtype=torch.uint8)
        tokens = torch.tensor([[2, 3, 0], [2, 4, 5]])
        proprio = torch.zeros((2, 10))
        delta = torch.zeros((2, 3))
        gripper = torch.tensor([1.0, 0.0])
        loss, _, _ = policy.loss(rgb, tokens, proprio, delta, gripper)
        loss.backward()
        self.assertEqual(policy(rgb, tokens, proprio)[0].shape, (2, 3))
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))

    def test_checkpoint_round_trip(self) -> None:
        policy = FlatMiniVLA(
            vocab_size=4,
            proprio_mean=np.zeros(10),
            proprio_std=np.ones(10),
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


if __name__ == "__main__":
    unittest.main()
