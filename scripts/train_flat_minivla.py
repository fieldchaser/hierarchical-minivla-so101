#!/usr/bin/env python3
"""Train Flat MiniVLA with a whole-episode language-stratified split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_minivla.flat_minivla import (
    FlatMiniVLA,
    build_vocabulary,
    encode_instructions,
    load_vision_episodes,
    save_flat_minivla,
    split_vision_episode_paths,
    transition_focused_mask,
)
from hierarchical_minivla.scripted_expert import PHASE_NAMES


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(
    arrays: dict[str, np.ndarray], vocabulary: dict[str, int]
) -> TensorDataset:
    tokens = encode_instructions(arrays["instruction"].tolist(), vocabulary)
    return TensorDataset(
        torch.from_numpy(arrays["rgb"]),
        torch.from_numpy(tokens),
        torch.from_numpy(arrays["proprio"]),
        torch.from_numpy(arrays["delta"]),
        torch.from_numpy(arrays["gripper_open"]),
    )


def move_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in batch)


@torch.inference_mode()
def evaluate(
    policy: FlatMiniVLA, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    policy.eval()
    total_loss = 0.0
    total_delta_error = 0.0
    total_gripper_correct = 0
    moving_delta_error = torch.zeros(3, dtype=torch.float64)
    moving_examples = 0
    total_examples = 0
    for batch in loader:
        rgb, tokens, proprio, delta, gripper_open = move_batch(batch, device)
        predicted_delta, gripper_logit = policy(rgb, tokens, proprio)
        normalized_prediction = (predicted_delta - policy.delta_mean) / policy.delta_std
        normalized_target = (delta - policy.delta_mean) / policy.delta_std
        loss = torch.nn.functional.mse_loss(
            normalized_prediction, normalized_target
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            gripper_logit, gripper_open
        )
        batch_size = len(rgb)
        total_loss += loss.item() * batch_size
        total_delta_error += (predicted_delta - delta).abs().sum().item()
        moving = torch.linalg.vector_norm(delta, dim=1) > 0.0005
        if moving.any():
            moving_delta_error += (
                (predicted_delta[moving] - delta[moving])
                .abs()
                .sum(dim=0)
                .double()
                .cpu()
            )
            moving_examples += int(moving.sum().item())
        total_gripper_correct += int(
            ((gripper_logit >= 0.0) == (gripper_open >= 0.5)).sum().item()
        )
        total_examples += batch_size
    result = {
        "loss": total_loss / total_examples,
        "delta_mae_mm": total_delta_error / (total_examples * 3) * 1000.0,
        "gripper_accuracy": total_gripper_correct / total_examples,
    }
    if moving_examples:
        moving_axis_mae = moving_delta_error / moving_examples * 1000.0
        result["moving_delta_mae_mm"] = float(moving_axis_mae.mean().item())
        result["moving_axis_mae_mm"] = moving_axis_mae.tolist()
    return result


def phase_counts(phases: np.ndarray) -> dict[str, int]:
    counts = np.bincount(phases, minlength=len(PHASE_NAMES))
    return {name: int(counts[index]) for index, name in enumerate(PHASE_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transition-focused", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/flat_minivla.pt")
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/flat_minivla_offline.json"),
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    paths = sorted(args.data_dir.expanduser().resolve().glob("episode_*.npz"))
    train_paths, validation_paths = split_vision_episode_paths(paths, seed=args.seed)
    full_train_arrays = load_vision_episodes(train_paths)
    full_validation_arrays = load_vision_episodes(validation_paths)
    train_arrays = full_train_arrays
    validation_arrays = full_validation_arrays
    if args.transition_focused:
        train_mask = transition_focused_mask(full_train_arrays)
        validation_mask = transition_focused_mask(full_validation_arrays)
        train_arrays = {
            key: values[train_mask] for key, values in full_train_arrays.items()
        }
        validation_arrays = {
            key: values[validation_mask]
            for key, values in full_validation_arrays.items()
        }
    vocabulary = build_vocabulary(train_arrays["instruction"].tolist())
    train_dataset = make_dataset(train_arrays, vocabulary)
    validation_dataset = make_dataset(validation_arrays, vocabulary)
    full_train_dataset = make_dataset(full_train_arrays, vocabulary)
    full_validation_dataset = make_dataset(full_validation_arrays, vocabulary)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    full_train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size)
    full_validation_loader = DataLoader(
        full_validation_dataset, batch_size=args.batch_size
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    policy = FlatMiniVLA(
        vocab_size=len(vocabulary),
        proprio_mean=train_arrays["proprio"].mean(axis=0),
        proprio_std=train_arrays["proprio"].std(axis=0),
        delta_mean=train_arrays["delta"].mean(axis=0),
        delta_std=train_arrays["delta"].std(axis=0),
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    best_epoch = 0
    best_validation = {"loss": float("inf")}
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_train_loss = 0.0
        for batch in train_loader:
            rgb, tokens, proprio, delta, gripper_open = move_batch(batch, device)
            loss, _, _ = policy.loss(rgb, tokens, proprio, delta, gripper_open)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(rgb)
        validation = evaluate(policy, validation_loader, device)
        train_loss = total_train_loss / len(train_dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, **validation})
        if validation["loss"] < best_validation["loss"]:
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in policy.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_delta_mae={validation['delta_mae_mm']:.3f}mm "
            f"val_gripper_acc={validation['gripper_accuracy']:.1%}"
        )

    assert best_state is not None
    policy.load_state_dict(best_state)
    train_metrics = evaluate(policy, train_eval_loader, device)
    validation_metrics = evaluate(policy, validation_loader, device)
    full_train_metrics = evaluate(policy, full_train_loader, device)
    full_validation_metrics = evaluate(policy, full_validation_loader, device)
    save_flat_minivla(policy, vocabulary, args.output.expanduser().resolve())
    result = {
        "device": str(device),
        "num_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "num_episodes": len(paths),
        "num_train_episodes": len(train_paths),
        "num_validation_episodes": len(validation_paths),
        "num_train_frames": len(train_dataset),
        "num_validation_frames": len(validation_dataset),
        "num_full_train_frames": len(full_train_dataset),
        "num_full_validation_frames": len(full_validation_dataset),
        "num_unique_train_instructions": len(set(train_arrays["instruction"])),
        "num_unique_validation_instructions": len(
            set(validation_arrays["instruction"])
        ),
        "train_phase_counts": phase_counts(train_arrays["phase"]),
        "validation_phase_counts": phase_counts(validation_arrays["phase"]),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "transition_focused": args.transition_focused,
        "train": train_metrics,
        "validation": validation_metrics,
        "full_train": full_train_metrics,
        "full_validation": full_validation_metrics,
        "history": history,
        "train_episodes": [path.name for path in train_paths],
        "validation_episodes": [path.name for path in validation_paths],
        "instruction_frame_counts": {
            str(instruction): int(count)
            for instruction, count in Counter(train_arrays["instruction"]).items()
        },
        "checkpoint": str(args.output),
    }
    metrics_path = args.metrics_output.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
