#!/usr/bin/env python3
"""Overfit one phase-balanced batch with Hierarchical MiniVLA."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from hierarchical_minivla.flat_minivla import (
    build_vocabulary,
    encode_instructions,
    load_vision_episodes,
    phase_balanced_indices,
)
from hierarchical_minivla.hierarchical_minivla import (
    HierarchicalMiniVLA,
    save_hierarchical_minivla,
)
from hierarchical_minivla.scripted_expert import PHASE_NAMES


@torch.inference_mode()
def metrics(
    policy: HierarchicalMiniVLA,
    rgb: torch.Tensor,
    tokens: torch.Tensor,
    proprio: torch.Tensor,
    delta: torch.Tensor,
    gripper_open: torch.Tensor,
    phase: torch.Tensor,
) -> dict[str, float]:
    policy.eval()
    predicted_delta, gripper_logit, phase_logits = policy(
        rgb, tokens, proprio, phase
    )
    return {
        "delta_mae_mm": (predicted_delta - delta).abs().mean().item() * 1000.0,
        "gripper_accuracy": (
            (gripper_logit >= 0.0) == (gripper_open >= 0.5)
        ).float().mean().item(),
        "phase_accuracy": (phase_logits.argmax(dim=1) == phase).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/hierarchical_minivla_overfit.pt"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/hierarchical_minivla_overfit.json"),
    )
    parser.add_argument("--max-delta-mae-mm", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(args.data_dir.expanduser().resolve().glob("episode_*.npz"))
    arrays = load_vision_episodes(paths)
    indices = phase_balanced_indices(arrays["phase"], args.batch_size)
    instructions = arrays["instruction"][indices].tolist()
    vocabulary = build_vocabulary(instructions)
    encoded = encode_instructions(instructions, vocabulary)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rgb = torch.as_tensor(arrays["rgb"][indices])
    tokens = torch.as_tensor(encoded, dtype=torch.long)
    proprio = torch.as_tensor(arrays["proprio"][indices], dtype=torch.float32)
    delta = torch.as_tensor(arrays["delta"][indices], dtype=torch.float32)
    gripper_open = torch.as_tensor(
        arrays["gripper_open"][indices], dtype=torch.float32
    )
    phase = torch.as_tensor(arrays["phase"][indices], dtype=torch.long)
    policy = HierarchicalMiniVLA(
        vocab_size=len(vocabulary),
        proprio_mean=proprio.mean(dim=0),
        proprio_std=proprio.std(dim=0),
        delta_mean=delta.mean(dim=0),
        delta_std=delta.std(dim=0),
        phase_weights=np.ones(len(PHASE_NAMES), dtype=np.float32),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    initial = metrics(
        policy, rgb, tokens, proprio, delta, gripper_open, phase
    )
    best = initial
    best_step = 0
    best_state = copy.deepcopy(policy.state_dict())
    for step in range(1, args.steps + 1):
        policy.train()
        loss, _, _, _ = policy.loss(
            rgb, tokens, proprio, delta, gripper_open, phase
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        should_check = step == 1 or step % 25 == 0 or step == args.steps
        if should_check:
            current = metrics(
                policy, rgb, tokens, proprio, delta, gripper_open, phase
            )
            current_rank = (
                current["phase_accuracy"],
                current["gripper_accuracy"],
                -current["delta_mae_mm"],
            )
            best_rank = (
                best["phase_accuracy"],
                best["gripper_accuracy"],
                -best["delta_mae_mm"],
            )
            if current_rank > best_rank:
                best = current
                best_step = step
                best_state = copy.deepcopy(policy.state_dict())
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"step={step:04d} loss={loss.item():.5f} "
                f"delta_mae={current['delta_mae_mm']:.3f}mm "
                f"gripper_acc={current['gripper_accuracy']:.1%} "
                f"phase_acc={current['phase_accuracy']:.1%}"
            )

    policy.load_state_dict(best_state)
    final = metrics(policy, rgb, tokens, proprio, delta, gripper_open, phase)
    save_hierarchical_minivla(
        policy, vocabulary, args.output.expanduser().resolve()
    )
    phase_counts = np.bincount(phase.numpy(), minlength=len(PHASE_NAMES))
    result = {
        "num_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "num_episodes": len(paths),
        "batch_size": len(indices),
        "steps": args.steps,
        "best_step": best_step,
        "proprio_dim": int(proprio.shape[1]),
        "vocabulary_size": len(vocabulary),
        "phase_counts": {
            name: int(phase_counts[index]) for index, name in enumerate(PHASE_NAMES)
        },
        "initial": initial,
        "final": final,
        "checkpoint": str(args.output),
    }
    metrics_path = args.metrics_output.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if final["delta_mae_mm"] > args.max_delta_mae_mm:
        raise SystemExit(
            f"Batch overfit MAE {final['delta_mae_mm']:.3f} mm exceeds "
            f"{args.max_delta_mae_mm:.3f} mm"
        )
    if final["gripper_accuracy"] < 1.0:
        raise SystemExit("Batch overfit did not reach 100% gripper accuracy")
    if final["phase_accuracy"] < 1.0:
        raise SystemExit("Batch overfit did not reach 100% phase accuracy")


if __name__ == "__main__":
    main()
