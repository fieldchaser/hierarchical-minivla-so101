from __future__ import annotations

import unittest

import numpy as np

from hierarchical_minivla.vision_data import (
    VISION_TEMPORAL_KEYS,
    validate_vision_episode_arrays,
)


def make_arrays(num_steps: int = 4) -> dict[str, np.ndarray]:
    arrays = {
        key: np.zeros((num_steps, 1), dtype=np.float32)
        for key in VISION_TEMPORAL_KEYS
    }
    arrays["rgb"] = np.zeros((num_steps, 8, 8, 3), dtype=np.uint8)
    arrays["instruction"] = np.asarray(["Pick up the red cube."] * num_steps)
    arrays["phase"] = np.zeros(num_steps, dtype=np.int64)
    return arrays


class VisionDataTest(unittest.TestCase):
    def test_valid_episode_reports_contract(self) -> None:
        summary = validate_vision_episode_arrays(make_arrays())
        self.assertEqual(summary["num_steps"], 4)
        self.assertEqual(summary["image_shape"], [8, 8, 3])
        self.assertEqual(summary["instruction"], "Pick up the red cube.")

    def test_misaligned_frames_are_rejected(self) -> None:
        arrays = make_arrays()
        arrays["rgb"] = arrays["rgb"][:-1]
        with self.assertRaisesRegex(ValueError, "not aligned"):
            validate_vision_episode_arrays(arrays)


if __name__ == "__main__":
    unittest.main()
