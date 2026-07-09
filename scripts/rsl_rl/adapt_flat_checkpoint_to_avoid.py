#!/usr/bin/env python3
"""Adapt a flat 29-DoF locomotion checkpoint for collision-avoidance warm start.

The adapter is intentionally file-local and non-invasive:
it reads an existing flat checkpoint and writes a new checkpoint whose first
actor/critic layer accepts the extra ball observations used by the avoidance
environment. Existing checkpoints are never modified.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch


DEFAULT_FLAT_CHECKPOINT = Path(
    "logs/rsl_rl/g1_29dof_flat_unitree_ppo/2026-04-24_13-47-26/model_6000.pt"
)
DEFAULT_OUTPUT_CHECKPOINT = Path(
    "logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_ppo/"
    "warmstart_from_unitree_flat_baseline_v4/model_0.pt"
)


def _expand_input_weight(
    state_dict: dict[str, torch.Tensor],
    key: str,
    target_input_dim: int,
) -> tuple[int, int]:
    """Expand a first-layer weight matrix by appending zeroed input columns."""
    if key not in state_dict:
        raise KeyError(f"Missing required tensor: {key}")

    source_weight = state_dict[key]
    if source_weight.ndim != 2:
        raise ValueError(f"{key} must be a 2-D weight tensor, got shape {tuple(source_weight.shape)}")

    output_dim, source_input_dim = source_weight.shape
    if target_input_dim < source_input_dim:
        raise ValueError(
            f"Target input dim for {key} ({target_input_dim}) is smaller than source dim "
            f"({source_input_dim})"
        )
    if target_input_dim == source_input_dim:
        return source_input_dim, target_input_dim

    target_weight = source_weight.new_zeros((output_dim, target_input_dim))
    target_weight[:, :source_input_dim] = source_weight
    state_dict[key] = target_weight
    return source_input_dim, target_input_dim


def _build_fresh_optimizer_state(
    optimizer_state_dict: dict[str, Any] | None,
    optimizer_lr: float | None,
) -> dict[str, Any]:
    """Keep optimizer hyperparameters but drop momentum tensors with old shapes."""
    if optimizer_state_dict is None:
        return {"state": {}, "param_groups": []}

    optimizer_state = copy.deepcopy(optimizer_state_dict)
    optimizer_state["state"] = {}

    if optimizer_lr is not None:
        for param_group in optimizer_state.get("param_groups", []):
            param_group["lr"] = optimizer_lr

    return optimizer_state


def adapt_checkpoint(args: argparse.Namespace) -> None:
    source_path = args.flat_checkpoint
    output_path = args.output_checkpoint

    if not source_path.is_file():
        raise FileNotFoundError(f"Flat checkpoint does not exist: {source_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    adapted = copy.deepcopy(checkpoint)

    actor_state_dict = adapted["actor_state_dict"]
    critic_state_dict = adapted["critic_state_dict"]

    actor_source_dim, actor_target_dim = _expand_input_weight(
        actor_state_dict,
        args.actor_first_layer_key,
        args.target_actor_input_dim,
    )
    critic_source_dim, critic_target_dim = _expand_input_weight(
        critic_state_dict,
        args.critic_first_layer_key,
        args.target_critic_input_dim,
    )

    adapted["optimizer_state_dict"] = _build_fresh_optimizer_state(
        adapted.get("optimizer_state_dict"),
        args.optimizer_lr,
    )
    adapted["iter"] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapted, output_path)

    print(f"[INFO] Source checkpoint: {source_path}")
    print(f"[INFO] Output checkpoint: {output_path}")
    print(f"[INFO] Actor input dim: {actor_source_dim} -> {actor_target_dim}")
    print(f"[INFO] Critic input dim: {critic_source_dim} -> {critic_target_dim}")
    print("[INFO] New observation weights were initialized to zero.")
    print("[INFO] Optimizer state was reset and training iteration was set to 0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a collision-avoidance warm-start checkpoint from a flat 29-DoF "
            "locomotion checkpoint."
        )
    )
    parser.add_argument(
        "--flat_checkpoint",
        type=Path,
        default=DEFAULT_FLAT_CHECKPOINT,
        help=(
            "Source flat locomotion checkpoint. The default matches the Unitree deploy "
            "baseline annotation."
        ),
    )
    parser.add_argument(
        "--output_checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT_CHECKPOINT,
        help="Destination checkpoint to create. Existing files are not overwritten by default.",
    )
    parser.add_argument(
        "--target_actor_input_dim",
        type=int,
        default=510,
        help="Avoidance actor input dimension. Default: 480 flat obs + 30 ball obs history.",
    )
    parser.add_argument(
        "--target_critic_input_dim",
        type=int,
        default=525,
        help="Avoidance critic input dimension. Default: 495 flat obs + 30 ball obs history.",
    )
    parser.add_argument(
        "--actor_first_layer_key",
        default="mlp.0.weight",
        help="Actor first linear layer key in the checkpoint state dict.",
    )
    parser.add_argument(
        "--critic_first_layer_key",
        default="mlp.0.weight",
        help="Critic first linear layer key in the checkpoint state dict.",
    )
    parser.add_argument(
        "--optimizer_lr",
        type=float,
        default=1.0e-3,
        help="Learning rate to write into the fresh optimizer param group.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the output checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    adapt_checkpoint(parse_args())
