#!/usr/bin/env python3
"""Train Hierarchical MiniVLA on whole episodes and report phase metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_minivla.flat_minivla import (
    build_vocabulary,
    encode_instructions,
    load_vision_episodes,
    split_vision_episode_paths,
)
from hierarchical_minivla.hierarchical_minivla import (
    HierarchicalMiniVLA,
    save_hierarchical_minivla,
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
        torch.from_numpy(arrays["phase"]),
    )


def move_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in batch)


def inverse_frequency_weights(phases: np.ndarray) -> np.ndarray:
    counts = np.bincount(phases, minlength=len(PHASE_NAMES)).astype(np.float32)
    if np.any(counts == 0):
        missing = [PHASE_NAMES[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"Training data is missing phases: {missing}")
    return counts.sum() / (len(PHASE_NAMES) * counts)


@torch.inference_mode()
def evaluate(
    policy: HierarchicalMiniVLA, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    policy.eval()
    num_phases = len(PHASE_NAMES)
    counts = torch.zeros(num_phases, dtype=torch.long)
    phase_correct = torch.zeros(num_phases, dtype=torch.long)
    teacher_delta_error = torch.zeros(num_phases, dtype=torch.float64)
    teacher_gripper_correct = torch.zeros(num_phases, dtype=torch.long)
    autonomous_delta_error = torch.zeros(num_phases, dtype=torch.float64)
    autonomous_gripper_correct = torch.zeros(num_phases, dtype=torch.long)
    normalized_delta_squared_error = 0.0
    gripper_bce_sum = 0.0
    phase_cross_entropy_sum = 0.0
    phase_weight_sum = 0.0

    for batch in loader:
        rgb, tokens, proprio, delta, gripper_open, phase = move_batch(batch, device)
        phase_logits, actions = policy._outputs(rgb, tokens, proprio)
        predicted_phase = phase_logits.argmax(dim=1)
        rows = torch.arange(len(rgb), device=device)
        teacher_actions = actions[rows, phase]
        autonomous_actions = actions[rows, predicted_phase]
        teacher_delta = teacher_actions[:, :3] * policy.delta_std + policy.delta_mean
        autonomous_delta = (
            autonomous_actions[:, :3] * policy.delta_std + policy.delta_mean
        )
        normalized_target = (delta - policy.delta_mean) / policy.delta_std
        normalized_delta_squared_error += F.mse_loss(
            teacher_actions[:, :3], normalized_target, reduction="sum"
        ).item()
        gripper_bce_sum += F.binary_cross_entropy_with_logits(
            teacher_actions[:, 3], gripper_open, reduction="sum"
        ).item()
        phase_cross_entropy_sum += F.cross_entropy(
            phase_logits,
            phase,
            weight=policy.phase_weights,
            reduction="sum",
        ).item()
        phase_weight_sum += policy.phase_weights[phase].sum().item()

        for phase_index in range(num_phases):
            mask = phase == phase_index
            if not mask.any():
                continue
            batch_count = int(mask.sum().item())
            counts[phase_index] += batch_count
            phase_correct[phase_index] += int(
                (predicted_phase[mask] == phase[mask]).sum().item()
            )
            teacher_delta_error[phase_index] += (
                (teacher_delta[mask] - delta[mask]).abs().sum().double().cpu()
            )
            teacher_gripper_correct[phase_index] += int(
                (
                    (teacher_actions[mask, 3] >= 0.0)
                    == (gripper_open[mask] >= 0.5)
                )
                .sum()
                .item()
            )
            autonomous_delta_error[phase_index] += (
                (autonomous_delta[mask] - delta[mask]).abs().sum().double().cpu()
            )
            autonomous_gripper_correct[phase_index] += int(
                (
                    (autonomous_actions[mask, 3] >= 0.0)
                    == (gripper_open[mask] >= 0.5)
                )
                .sum()
                .item()
            )

    total_examples = int(counts.sum().item())
    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    delta_mse = normalized_delta_squared_error / (total_examples * 3)
    gripper_loss = gripper_bce_sum / total_examples
    phase_loss = phase_cross_entropy_sum / phase_weight_sum

    def routed_metrics(
        delta_error: torch.Tensor, gripper_correct: torch.Tensor
    ) -> dict[str, float]:
        return {
            "delta_mae_mm": float(
                delta_error.sum().item() / (total_examples * 3) * 1000.0
            ),
            "gripper_accuracy": float(
                gripper_correct.sum().item() / total_examples
            ),
        }

    per_phase = {}
    for index, name in enumerate(PHASE_NAMES):
        count = int(counts[index].item())
        per_phase[name] = {
            "count": count,
            "phase_accuracy": float(phase_correct[index].item() / count),
            "teacher_delta_mae_mm": float(
                teacher_delta_error[index].item() / (count * 3) * 1000.0
            ),
            "teacher_gripper_accuracy": float(
                teacher_gripper_correct[index].item() / count
            ),
            "autonomous_delta_mae_mm": float(
                autonomous_delta_error[index].item() / (count * 3) * 1000.0
            ),
            "autonomous_gripper_accuracy": float(
                autonomous_gripper_correct[index].item() / count
            ),
        }

    return {
        "loss": delta_mse + gripper_loss + phase_loss,
        "delta_loss": delta_mse,
        "gripper_loss": gripper_loss,
        "phase_loss": phase_loss,
        "phase_accuracy": float(phase_correct.sum().item() / total_examples),
        "teacher_routed": routed_metrics(
            teacher_delta_error, teacher_gripper_correct
        ),
        "autonomous_routed": routed_metrics(
            autonomous_delta_error, autonomous_gripper_correct
        ),
        "per_phase": per_phase,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--extra-train-dir", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--dagger-train-dir", type=Path, action="append", default=[]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/hierarchical_minivla.pt")
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/hierarchical_minivla_offline.json"),
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    paths = sorted(args.data_dir.expanduser().resolve().glob("episode_*.npz"))
    base_train_paths, validation_paths = split_vision_episode_paths(
        paths, seed=args.seed
    )
    extra_train_paths = []
    for extra_train_dir in args.extra_train_dir:
        directory_paths = sorted(
            extra_train_dir.expanduser().resolve().glob("episode_*.npz")
        )
        if not directory_paths:
            parser.error("--extra-train-dir contains no episode_*.npz files")
        extra_train_paths.extend(directory_paths)
    dagger_train_paths = []
    for dagger_train_dir in args.dagger_train_dir:
        directory_paths = sorted(
            dagger_train_dir.expanduser().resolve().glob("episode_*.npz")
        )
        if not directory_paths:
            parser.error("--dagger-train-dir contains no episode_*.npz files")
        dagger_train_paths.extend(directory_paths)
    additional_paths = extra_train_paths + dagger_train_paths
    if len(set(additional_paths)) != len(additional_paths):
        parser.error("Additional training directories contain duplicate episodes")
    overlap = set(paths) & set(additional_paths)
    if overlap:
        parser.error("Additional training directories overlap the base dataset")
    full_train_paths = base_train_paths + extra_train_paths
    train_paths = full_train_paths + dagger_train_paths
    train_arrays = load_vision_episodes(full_train_paths)
    num_dagger_train_frames = 0
    if dagger_train_paths:
        dagger_arrays = load_vision_episodes(
            dagger_train_paths, frame_mask_key="dagger"
        )
        num_dagger_train_frames = len(dagger_arrays["rgb"])
        train_arrays = {
            key: np.concatenate([train_arrays[key], dagger_arrays[key]])
            for key in train_arrays
        }
    validation_arrays = load_vision_episodes(validation_paths)
    vocabulary = build_vocabulary(train_arrays["instruction"].tolist())
    train_dataset = make_dataset(train_arrays, vocabulary)
    validation_dataset = make_dataset(validation_arrays, vocabulary)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    phase_weights = inverse_frequency_weights(train_arrays["phase"])
    policy = HierarchicalMiniVLA(
        vocab_size=len(vocabulary),
        proprio_mean=train_arrays["proprio"].mean(axis=0),
        proprio_std=train_arrays["proprio"].std(axis=0),
        delta_mean=train_arrays["delta"].mean(axis=0),
        delta_std=train_arrays["delta"].std(axis=0),
        phase_weights=phase_weights,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_train_loss = 0.0
        for batch in train_loader:
            rgb, tokens, proprio, delta, gripper_open, phase = move_batch(
                batch, device
            )
            loss, _, _, _ = policy.loss(
                rgb, tokens, proprio, delta, gripper_open, phase
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(rgb)
        validation = evaluate(policy, validation_loader, device)
        train_loss = total_train_loss / len(train_dataset)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_phase_accuracy": validation["phase_accuracy"],
                "validation_teacher_delta_mae_mm": validation["teacher_routed"][
                    "delta_mae_mm"
                ],
                "validation_autonomous_delta_mae_mm": validation[
                    "autonomous_routed"
                ]["delta_mae_mm"],
            }
        )
        if validation["loss"] < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = float(validation["loss"])
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in policy.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"phase_acc={validation['phase_accuracy']:.1%} "
            f"teacher_delta={validation['teacher_routed']['delta_mae_mm']:.3f}mm "
            f"auto_delta={validation['autonomous_routed']['delta_mae_mm']:.3f}mm"
        )

    assert best_state is not None
    policy.load_state_dict(best_state)
    train_metrics = evaluate(policy, train_eval_loader, device)
    validation_metrics = evaluate(policy, validation_loader, device)
    save_hierarchical_minivla(
        policy, vocabulary, args.output.expanduser().resolve()
    )
    phase_counts = np.bincount(
        train_arrays["phase"], minlength=len(PHASE_NAMES)
    )
    result = {
        "device": str(device),
        "num_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "num_base_episodes": len(paths),
        "num_base_train_episodes": len(base_train_paths),
        "num_extra_train_episodes": len(additional_paths),
        "num_full_extra_train_episodes": len(extra_train_paths),
        "num_dagger_train_episodes": len(dagger_train_paths),
        "num_dagger_train_frames": num_dagger_train_frames,
        "num_train_episodes": len(train_paths),
        "num_validation_episodes": len(validation_paths),
        "num_train_frames": len(train_dataset),
        "num_validation_frames": len(validation_dataset),
        "num_unique_train_instructions": len(set(train_arrays["instruction"])),
        "num_unique_validation_instructions": len(
            set(validation_arrays["instruction"])
        ),
        "phase_weights": {
            name: float(phase_weights[index])
            for index, name in enumerate(PHASE_NAMES)
        },
        "train_phase_counts": {
            name: int(phase_counts[index])
            for index, name in enumerate(PHASE_NAMES)
        },
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train": train_metrics,
        "validation": validation_metrics,
        "history": history,
        "train_episodes": [path.name for path in train_paths],
        "extra_train_episodes": [path.name for path in additional_paths],
        "full_extra_train_episodes": [path.name for path in extra_train_paths],
        "dagger_train_episodes": [path.name for path in dagger_train_paths],
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
