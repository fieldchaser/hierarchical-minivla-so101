from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from hierarchical_minivla.hierarchical_minivla import (
    HierarchicalMiniVLA,
    load_hierarchical_minivla,
    save_hierarchical_minivla,
)


def make_policy() -> HierarchicalMiniVLA:
    return HierarchicalMiniVLA(
        vocab_size=6,
        proprio_mean=np.zeros(13),
        proprio_std=np.ones(13),
        delta_mean=np.zeros(3),
        delta_std=np.ones(3),
        phase_weights=np.ones(6),
    )


class HierarchicalMiniVLATest(unittest.TestCase):
    def test_forward_and_backward_use_phase_specific_heads(self) -> None:
        policy = make_policy()
        rgb = torch.zeros((3, 32, 32, 3), dtype=torch.uint8)
        tokens = torch.tensor([[2, 3], [2, 4], [2, 5]])
        proprio = torch.zeros((3, 13))
        phases = torch.tensor([0, 2, 5])
        loss, _, _, _ = policy.loss(
            rgb,
            tokens,
            proprio,
            torch.zeros((3, 3)),
            torch.tensor([1.0, 0.0, 1.0]),
            phases,
        )
        loss.backward()
        delta, gripper, phase_logits = policy(rgb, tokens, proprio, phases)
        self.assertEqual(delta.shape, (3, 3))
        self.assertEqual(gripper.shape, (3,))
        self.assertEqual(phase_logits.shape, (3, 6))
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))

    def test_checkpoint_round_trip(self) -> None:
        policy = make_policy()
        vocabulary = {"<pad>": 0, "<unk>": 1, "pick": 2, "red": 3, "cube": 4, "bin": 5}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hierarchical.pt"
            save_hierarchical_minivla(policy, vocabulary, path)
            restored, restored_vocabulary = load_hierarchical_minivla(path)
        self.assertEqual(restored_vocabulary, vocabulary)
        for expected, actual in zip(policy.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
