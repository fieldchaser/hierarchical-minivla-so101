#!/usr/bin/env python3
"""Train and evaluate the milestone-0 behavior-cloning baseline."""

from __future__ import annotations

import argparse
import json

import numpy as np

from hierarchical_minivla.baseline import GoalConditionedRidgePolicy, make_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_states, train_actions = make_dataset(args.train_samples, seed=args.seed)
    test_states, test_actions = make_dataset(args.test_samples, seed=args.seed + 1)
    policy = GoalConditionedRidgePolicy().fit(train_states, train_actions)
    predictions = policy.predict(test_states)

    mse = float(np.mean((predictions - test_actions) ** 2))
    goal_probe = np.repeat(test_states[:1], 3, axis=0)
    goal_probe[:, 13:16] = np.eye(3)
    probe_actions = policy.predict(goal_probe)[:, :3]

    print(
        json.dumps(
            {
                "train_samples": args.train_samples,
                "test_samples": args.test_samples,
                "test_mse": mse,
                "goal_probe_delta_xyz": probe_actions.round(5).tolist(),
            },
            indent=2,
        )
    )
    if mse >= 1e-8:
        raise SystemExit(f"Smoke baseline MSE is unexpectedly high: {mse:.3e}")


if __name__ == "__main__":
    main()

