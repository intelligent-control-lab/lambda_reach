"""Export a trained safety value function to ONNX for unitree_rl_lab deploy."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_value.safety_analysis_algos.model import build_mlp  # noqa: E402


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


def verify_onnx(path: Path, input_name: str, expected_input_dim: int) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    if not model.graph.input:
        raise RuntimeError("Exported ONNX model has no graph inputs.")

    graph_input = model.graph.input[0]
    dims = graph_input.type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in dims]
    if graph_input.name != input_name:
        raise RuntimeError(f"Expected ONNX input name '{input_name}', got '{graph_input.name}'")
    if input_shape != [1, expected_input_dim]:
        raise RuntimeError(f"Expected ONNX input shape [1, {expected_input_dim}], got {input_shape}")

    output_shape = []
    if model.graph.output:
        output_dims = model.graph.output[0].type.tensor_type.shape.dim
        output_shape = [dim.dim_value for dim in output_dims]

    print(f"[OK] ONNX input name: {graph_input.name}")
    print(f"[OK] ONNX input shape: {input_shape}")
    print(f"[OK] ONNX output name: {model.graph.output[0].name if model.graph.output else '<none>'}")
    print(f"[OK] ONNX output shape: {output_shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export safety value function to ONNX.")
    parser.add_argument("--safety_value_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input_name", type=str, default="obs")
    parser.add_argument("--output_name", type=str, default="safety_value")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, input_dim, training_config, checkpoint_path = load_model(args.safety_value_dir, args.checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy_obs = torch.zeros(1, input_dim, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_obs,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=[args.input_name],
            output_names=[args.output_name],
            dynamic_axes=None,
        )

    verify_onnx(args.output, args.input_name, input_dim)
    print(f"[OK] Exported safety value ONNX: {args.output}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Algorithm: {training_config.get('algorithm')}")
    print(f"[INFO] Hidden dims: {training_config.get('hidden_dims')}")


if __name__ == "__main__":
    main()
