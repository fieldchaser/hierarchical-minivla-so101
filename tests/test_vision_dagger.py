from __future__ import annotations

import unittest

from hierarchical_minivla.vision_dagger import run_visual_dagger_episode


class VisionDaggerTest(unittest.TestCase):
    def test_negative_learner_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "learner_reach_steps"):
            run_visual_dagger_episode(
                None,
                None,
                None,
                "instruction",
                None,
                learner_reach_steps=-1,
            )

    def test_nonpositive_transition_votes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "transition_votes"):
            run_visual_dagger_episode(
                None,
                None,
                None,
                "instruction",
                None,
                transition_votes=0,
            )


if __name__ == "__main__":
    unittest.main()
