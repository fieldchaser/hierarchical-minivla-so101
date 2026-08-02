#!/usr/bin/env python3
"""Train the neural state-and-one-hot-goal behavior-cloning baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_minivla.state_policy import (
    StateGoalPolicy,
    load_episodes,
    save_policy,
    split_episode_paths,
)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    policy: StateGoalPolicy,
    features: torch.Tensor,
    deltas: torch.Tensor,
    gripper_open: torch.Tensor,
) -> dict[str, float]:
    policy.eval()
    predicted_delta, gripper_logit = policy(features)
    loss, delta_loss, gripper_loss = policy.loss(features, deltas, gripper_open)
    return {
        "loss": loss.item(),
        "delta_loss": delta_loss.item(),
        "gripper_loss": gripper_loss.item(),
        "delta_mae_mm": (predicted_delta - deltas).abs().mean().item() * 1000.0,
        "gripper_accuracy": (
            (gripper_logit >= 0.0) == (gripper_open >= 0.5)
        ).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/state_goal_policy.pt")
    )
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    args = parser.parse_args()

    paths = sorted(args.data_dir.expanduser().resolve().glob("episode_*.npz"))
    train_paths, validation_paths = split_episode_paths(paths, seed=args.seed)
    train_features, train_deltas, train_gripper = load_episodes(train_paths)
    validation_features, validation_deltas, validation_gripper = load_episodes(
        validation_paths
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    policy = StateGoalPolicy(
        feature_mean=train_features.mean(axis=0),
        feature_std=train_features.std(axis=0),
        delta_mean=train_deltas.mean(axis=0),
        delta_std=train_deltas.std(axis=0),
        hidden_dim=args.hidden_dim,
    ).to(device)

    train_tensors = tuple(
        torch.as_tensor(array, dtype=torch.float32) for array in (
            train_features,
            train_deltas,
            train_gripper,
        )
    )
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_tensors = tuple(
        torch.as_tensor(array, dtype=torch.float32, device=device)
        for array in (validation_features, validation_deltas, validation_gripper)
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        policy.train()
        for features, deltas, gripper_open in loader:
            features = features.to(device)
            deltas = deltas.to(device)
            gripper_open = gripper_open.to(device)
            loss, _, _ = policy.loss(features, deltas, gripper_open)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate(policy, *validation_tensors)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} val_loss={metrics['loss']:.4f} "
                f"delta_mae={metrics['delta_mae_mm']:.3f}mm "
                f"gripper_acc={metrics['gripper_accuracy']:.1%}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    policy.load_state_dict(best_state)
    final_metrics = evaluate(policy, *validation_tensors)
    save_policy(policy, args.output.expanduser().resolve())
    result = {
        "device": str(device),
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "train_steps": int(train_features.shape[0]),
        "validation_steps": int(validation_features.shape[0]),
        "best_epoch": best_epoch,
        **final_metrics,
        "checkpoint": str(args.output),
    }
    if args.metrics_output:
        metrics_path = args.metrics_output.expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
