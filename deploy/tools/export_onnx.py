"""Export a trained safety value function to ONNX for unitree_rl_lab deploy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_value.monitoring.value_model import (  # noqa: E402
    LambdaValueEnsemble, SingleValueWrapper, resolve_checkpoint, load_model,
)


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
            dynamo=False,  # Preserve the original exporter across PyTorch 2.9+ defaults.
            do_constant_folding=True,
            input_names=[args.input_name],
            output_names=[args.output_name],
            dynamic_axes=None,
        )

    import onnx
    exported = onnx.load(str(args.output))
    onnx.helper.set_model_props(exported, {"lambda.training_config": json.dumps(training_config)})
    onnx.save(exported, str(args.output))
    verify_onnx(args.output, args.input_name, input_dim)
    print(f"[OK] Exported safety value ONNX: {args.output}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Algorithm: {training_config.get('algorithm')}")
    print(f"[INFO] Hidden dims: {training_config.get('hidden_dims')}")


if __name__ == "__main__":
    main()
