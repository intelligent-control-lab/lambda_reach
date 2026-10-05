"""Shared checkpoint loading for export and simulator-independent monitoring."""

import glob
import json
import os
from pathlib import Path
import torch
from safety_value.safety_analysis_algos.model import build_mlp

class LambdaValueEnsemble(torch.nn.Module):
    """Two-critic lambda-reachability value function exported as one scalar output."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.critic1 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)
        self.critic2 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        value = 0.5 * (self.critic1(obs) + self.critic2(obs))
        return value.reshape(obs.shape[0], 1)


class SingleValueWrapper(torch.nn.Module):
    """Normalize single-critic value output to shape [batch, 1]."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        value = self.model(obs)
        return value.reshape(obs.shape[0], 1)


def resolve_checkpoint(models_dir: Path) -> Path:
    """Match inference recorder checkpoint priority."""
    step_ckpts = glob.glob(str(models_dir / "step_*.pt"))
    step_entries: list[tuple[int, str]] = []
    for ckpt_path in step_ckpts:
        basename = os.path.basename(ckpt_path)
        try:
            step = int(basename.split("_")[1].replace(".pt", ""))
        except (IndexError, ValueError):
            continue
        step_entries.append((step, ckpt_path))
    if step_entries:
        return Path(max(step_entries, key=lambda item: item[0])[1])

    seed_ckpts = glob.glob(str(models_dir / "last_seed_*.pt"))
    seed_entries: list[tuple[int, str]] = []
    for ckpt_path in seed_ckpts:
        basename = os.path.basename(ckpt_path)
        try:
            seed = int(basename.split("_")[2].replace(".pt", ""))
        except (IndexError, ValueError):
            continue
        seed_entries.append((seed, ckpt_path))
    if seed_entries:
        return Path(sorted(seed_entries, key=lambda item: item[0])[0][1])

    last_ckpt = models_dir / "last.pt"
    if last_ckpt.exists():
        return last_ckpt

    raise FileNotFoundError(f"No safety value checkpoints found in {models_dir}")


def load_model(safety_value_dir: Path, checkpoint_path: Path | None) -> tuple[torch.nn.Module, int, dict, Path]:
    config_path = safety_value_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training_config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        training_config = json.load(f)

    input_dim = int(training_config["input_dim"])
    hidden_dims = tuple(int(dim) for dim in training_config.get("hidden_dims", [256, 256]))
    algorithm = training_config.get("algorithm", "dpe")

    if checkpoint_path is None:
        checkpoint_path = resolve_checkpoint(safety_value_dir / "models")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if algorithm == "lambda_reachability":
        model = LambdaValueEnsemble(input_dim=input_dim, hidden_dims=hidden_dims)
        model.critic1.load_state_dict(ckpt["critic1"])
        model.critic2.load_state_dict(ckpt["critic2"])
    else:
        base_model = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)
        base_model.load_state_dict(ckpt["model_state"])
        model = SingleValueWrapper(base_model)

    model.eval()
    return model, input_dim, training_config, checkpoint_path
