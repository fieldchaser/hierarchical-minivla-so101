#!/usr/bin/env python3
"""Train the phase-aware state-and-goal behavior-cloning policy."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_minivla.hierarchical_policy import (
    HierarchicalStateGoalPolicy,
    balanced_phase_weights,
    load_hierarchical_episodes,
    save_hierarchical_policy,
)
from hierarchical_minivla.scripted_expert import PHASE_NAMES
from hierarchical_minivla.state_policy import split_episode_paths


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
    policy: HierarchicalStateGoalPolicy,
    features: torch.Tensor,
    deltas: torch.Tensor,
    gripper_open: torch.Tensor,
    phases: torch.Tensor,
) -> dict[str, float | list[float]]:
    policy.eval()
    predicted_delta, gripper_logit, phase_logits = policy(features, phases)
    loss, delta_loss, gripper_loss, phase_loss = policy.loss(
        features, deltas, gripper_open, phases
    )
    predicted_phases = phase_logits.argmax(dim=1)
    per_phase_accuracy = []
    for phase in range(len(PHASE_NAMES)):
        mask = phases == phase
        accuracy = (predicted_phases[mask] == phases[mask]).float().mean().item()
        per_phase_accuracy.append(accuracy)
    return {
        "loss": loss.item(),
        "delta_loss": delta_loss.item(),
        "gripper_loss": gripper_loss.item(),
        "phase_loss": phase_loss.item(),
        "delta_mae_mm": (predicted_delta - deltas).abs().mean().item() * 1000.0,
        "gripper_accuracy": (
            (gripper_logit >= 0.0) == (gripper_open >= 0.5)
        ).float().mean().item(),
        "phase_accuracy": (predicted_phases == phases).float().mean().item(),
        "per_phase_accuracy": per_phase_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/hierarchical_state_policy.pt")
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

    paths = sorted(
        path
        for data_dir in args.data_dir
        for path in data_dir.expanduser().resolve().glob("episode_*.npz")
    )
    train_paths, validation_paths = split_episode_paths(paths, seed=args.seed)
    train_arrays = load_hierarchical_episodes(train_paths)
    validation_arrays = load_hierarchical_episodes(validation_paths)
    train_features, train_deltas, train_gripper, train_phases = train_arrays

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    policy = HierarchicalStateGoalPolicy(
        feature_mean=train_features.mean(axis=0),
        feature_std=train_features.std(axis=0),
        delta_mean=train_deltas.mean(axis=0),
        delta_std=train_deltas.std(axis=0),
        phase_weights=balanced_phase_weights(train_phases),
        hidden_dim=args.hidden_dim,
    ).to(device)

    train_tensors = (
        torch.as_tensor(train_features, dtype=torch.float32),
        torch.as_tensor(train_deltas, dtype=torch.float32),
        torch.as_tensor(train_gripper, dtype=torch.float32),
        torch.as_tensor(train_phases, dtype=torch.long),
    )
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_tensors = (
        torch.as_tensor(validation_arrays[0], dtype=torch.float32, device=device),
        torch.as_tensor(validation_arrays[1], dtype=torch.float32, device=device),
        torch.as_tensor(validation_arrays[2], dtype=torch.float32, device=device),
        torch.as_tensor(validation_arrays[3], dtype=torch.long, device=device),
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        policy.train()
        for features, deltas, gripper_open, phases in loader:
            features = features.to(device)
            deltas = deltas.to(device)
            gripper_open = gripper_open.to(device)
            phases = phases.to(device)
            loss, _, _, _ = policy.loss(features, deltas, gripper_open, phases)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate(policy, *validation_tensors)
        if metrics["loss"] < best_loss:
            best_loss = float(metrics["loss"])
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} val_loss={metrics['loss']:.4f} "
                f"delta_mae={metrics['delta_mae_mm']:.3f}mm "
                f"phase_acc={metrics['phase_accuracy']:.1%}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    policy.load_state_dict(best_state)
    final_metrics = evaluate(policy, *validation_tensors)
    save_hierarchical_policy(policy, args.output.expanduser().resolve())
    result = {
        "device": str(device),
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "train_steps": int(train_features.shape[0]),
        "validation_steps": int(validation_arrays[0].shape[0]),
        "best_epoch": best_epoch,
        **final_metrics,
        "phase_names": list(PHASE_NAMES),
        "checkpoint": str(args.output),
    }
    if args.metrics_output:
        metrics_path = args.metrics_output.expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
