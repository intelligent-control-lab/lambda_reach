#!/usr/bin/env python3
"""Offline loader and model inference for G1 real-robot velocity logs.

The real-robot logger saves each experiment run as:

    <run_dir>/metadata.yaml
    <run_dir>/records.csv
    <run_dir>/context_raw.csv

This script reads records.csv, extracts obs_0..obs_479, and runs the
deployment policy and safety-value models on the recorded trajectory.

Examples:

    # Use model paths recorded in metadata.yaml when possible. In env_isaaclab,
    # auto falls back to the local PyTorch ckpts if onnxruntime is unavailable.
    python scripts/real_exp_offline_inference.py

    # Use explicit deployment ONNX models.
    python scripts/real_exp_offline_inference.py \
        --policy-model /path/to/policy.onnx \
        --safety-value-model /path/to/safety_value.onnx

    # Use a raw RSL-RL actor checkpoint and a DPE safety-value checkpoint.
    python scripts/real_exp_offline_inference.py \
        --policy-model logs/rsl_rl/.../model_6000.pt \
        --safety-value-model logs/safety_analysis/.../models/last_seed_0.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment check
    raise SystemExit(
        "Missing dependency: numpy. Run this script inside the project/IsaacLab "
        "Python environment, or install numpy first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "logs" / "real_exp"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "real_exp_offline_inference"

OBS_DIM = 480
ACTION_DIM = 29
COMMAND_DIM = 3

OBS_COLUMNS = [f"obs_{idx}" for idx in range(OBS_DIM)]
COMMAND_COLUMNS = [f"command_{idx}" for idx in range(COMMAND_DIM)]
PREV_ACTION_COLUMNS = [f"prev_action_{idx}" for idx in range(ACTION_DIM)]
POLICY_ACTION_COLUMNS = [f"policy_action_{idx}" for idx in range(ACTION_DIM)]
TARGET_Q_COLUMNS = [f"target_q_{idx}" for idx in range(ACTION_DIM)]

SCALAR_COLUMNS = [
    "time_s",
    "relative_time_s",
    "control_step",
    "dt_actual",
    "safety_valid",
    "safety_value",
    "safety_signal",
    "safety_inference_ms",
    "policy_inference_ms",
]

DEPLOY_POLICY_FALLBACKS: list[Path] = []

DEPLOY_SAFETY_FALLBACKS: list[Path] = []


@dataclass(frozen=True)
class RealExpRun:
    """One real-robot experiment run."""

    run_dir: Path
    records_csv: Path
    context_csv: Path | None
    metadata_yaml: Path | None
    rel_name: str


@dataclass
class RealExpRecords:
    """Loaded numeric arrays from records.csv."""

    time_s: np.ndarray
    relative_time_s: np.ndarray
    control_step: np.ndarray
    dt_actual: np.ndarray
    command: np.ndarray
    obs: np.ndarray
    safety_valid: np.ndarray
    safety_value: np.ndarray
    safety_signal: np.ndarray
    safety_inference_ms: np.ndarray
    prev_action: np.ndarray
    policy_action: np.ndarray
    target_q: np.ndarray
    policy_inference_ms: np.ndarray


class ModelRunner:
    """Small common interface for torch / onnx model wrappers."""

    path: Path

    def predict(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def discover_runs(data_root: Path, run_filter: str | None = None) -> list[RealExpRun]:
    """Return all run directories under data_root that contain records.csv."""

    data_root = data_root.resolve()
    runs: list[RealExpRun] = []
    for records_csv in sorted(data_root.rglob("records.csv")):
        run_dir = records_csv.parent
        rel_name = run_dir.relative_to(data_root).as_posix()
        if run_filter and run_filter not in rel_name:
            continue
        context_csv = run_dir / "context_raw.csv"
        metadata_yaml = run_dir / "metadata.yaml"
        runs.append(
            RealExpRun(
                run_dir=run_dir,
                records_csv=records_csv,
                context_csv=context_csv if context_csv.exists() else None,
                metadata_yaml=metadata_yaml if metadata_yaml.exists() else None,
                rel_name=rel_name,
            )
        )
    return runs


def read_metadata_values(metadata_yaml: Path | None) -> dict[str, str]:
    """Read simple top-level key: value pairs from metadata.yaml.

    The deployed metadata files are intentionally simple, so this avoids a
    hard dependency on PyYAML for an offline utility script.
    """

    if metadata_yaml is None or not metadata_yaml.exists():
        return {}

    values: dict[str, str] = {}
    for line in metadata_yaml.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def resolve_model_arg(
    model_arg: str,
    data_root: Path,
    metadata_key: str,
    fallbacks: Iterable[Path],
) -> Path | None:
    """Resolve a model CLI argument.

    "auto" means: first use metadata.yaml, then fall back to known local paths.
    "none" disables that model.
    """

    normalized = model_arg.strip().lower()
    if normalized == "none":
        return None
    if normalized != "auto":
        return Path(model_arg).expanduser().resolve()

    for run in discover_runs(data_root):
        metadata = read_metadata_values(run.metadata_yaml)
        raw_path = metadata.get(metadata_key)
        if raw_path:
            path = Path(raw_path).expanduser()
            if path.exists() and _auto_model_candidate(path):
                return path.resolve()

    for fallback in fallbacks:
        if fallback.exists() and _auto_model_candidate(fallback):
            return fallback.resolve()

    return None


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _auto_model_candidate(path: Path) -> bool:
    """Return True if an auto-resolved model can be loaded in this environment."""

    return path.suffix.lower() != ".onnx" or _onnxruntime_available()


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="") as f:
        return next(csv.reader(f))


def _column_indices(header: list[str], columns: Iterable[str]) -> list[int]:
    index = {name: idx for idx, name in enumerate(header)}
    missing = [name for name in columns if name not in index]
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"Missing columns in records.csv: {preview}")
    return [index[name] for name in columns]


def _load_numeric_columns(
    csv_path: Path,
    header: list[str],
    columns: list[str],
    dtype: np.dtype | type = np.float32,
    max_rows: int | None = None,
) -> np.ndarray:
    usecols = _column_indices(header, columns)
    data = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=usecols,
        dtype=dtype,
        ndmin=2,
        max_rows=max_rows,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_real_exp_records(records_csv: Path, max_rows: int | None = None) -> RealExpRecords:
    """Load all columns needed for offline trajectory inference."""

    header = _read_header(records_csv)
    columns = [
        *SCALAR_COLUMNS,
        *COMMAND_COLUMNS,
        *OBS_COLUMNS,
        *PREV_ACTION_COLUMNS,
        *POLICY_ACTION_COLUMNS,
        *TARGET_Q_COLUMNS,
    ]
    values = _load_numeric_columns(records_csv, header, columns, max_rows=max_rows)

    offset = 0
    scalars = values[:, offset : offset + len(SCALAR_COLUMNS)]
    offset += len(SCALAR_COLUMNS)
    command = values[:, offset : offset + len(COMMAND_COLUMNS)]
    offset += len(COMMAND_COLUMNS)
    obs = values[:, offset : offset + len(OBS_COLUMNS)]
    offset += len(OBS_COLUMNS)
    prev_action = values[:, offset : offset + len(PREV_ACTION_COLUMNS)]
    offset += len(PREV_ACTION_COLUMNS)
    policy_action = values[:, offset : offset + len(POLICY_ACTION_COLUMNS)]
    offset += len(POLICY_ACTION_COLUMNS)
    target_q = values[:, offset : offset + len(TARGET_Q_COLUMNS)]

    return RealExpRecords(
        time_s=scalars[:, 0].astype(np.float64, copy=False),
        relative_time_s=scalars[:, 1].astype(np.float64, copy=False),
        control_step=scalars[:, 2].astype(np.int64, copy=False),
        dt_actual=scalars[:, 3].astype(np.float64, copy=False),
        command=command,
        obs=obs,
        safety_valid=scalars[:, 4].astype(np.bool_, copy=False),
        safety_value=scalars[:, 5],
        safety_signal=scalars[:, 6],
        safety_inference_ms=scalars[:, 7],
        prev_action=prev_action,
        policy_action=policy_action,
        target_q=target_q,
        policy_inference_ms=scalars[:, 8],
    )


def load_context_raw(context_csv: Path, max_rows: int | None = None) -> dict[str, np.ndarray]:
    """Load context_raw.csv state arrays for pre/post velocity analysis."""

    header = _read_header(context_csv)
    numeric_columns = [
        "time_s",
        "relative_time_s",
        *[f"root_ang_vel_{idx}" for idx in range(3)],
        *[f"root_quat_{idx}" for idx in range(4)],
        *[f"projected_gravity_{idx}" for idx in range(3)],
        *[f"joint_pos_{idx}" for idx in range(ACTION_DIM)],
        *[f"joint_vel_{idx}" for idx in range(ACTION_DIM)],
    ]
    numeric = _load_numeric_columns(context_csv, header, numeric_columns, max_rows=max_rows)

    phases: list[str] = []
    fsm_states: list[str] = []
    phase_idx = _column_indices(header, ["phase"])[0]
    fsm_idx = _column_indices(header, ["fsm_state"])[0]
    with context_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row_count, row in enumerate(reader):
            if max_rows is not None and row_count >= max_rows:
                break
            phases.append(row[phase_idx])
            fsm_states.append(row[fsm_idx])

    offset = 0
    result = {
        "time_s": numeric[:, offset].astype(np.float64, copy=False),
        "relative_time_s": numeric[:, offset + 1].astype(np.float64, copy=False),
        "phase": np.asarray(phases),
        "fsm_state": np.asarray(fsm_states),
    }
    offset += 2
    result["root_ang_vel"] = numeric[:, offset : offset + 3]
    offset += 3
    result["root_quat"] = numeric[:, offset : offset + 4]
    offset += 4
    result["projected_gravity"] = numeric[:, offset : offset + 3]
    offset += 3
    result["joint_pos"] = numeric[:, offset : offset + ACTION_DIM]
    offset += ACTION_DIM
    result["joint_vel"] = numeric[:, offset : offset + ACTION_DIM]
    return result


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment check
        raise SystemExit(
            "Missing dependency: torch. Run inside the project/IsaacLab Python "
            "environment when using .pt models."
        ) from exc
    return torch


def _torch_load(path: Path, map_location: str):
    torch = _import_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _activation_module(name: str):
    torch = _import_torch()
    normalized = name.lower()
    if normalized == "elu":
        return torch.nn.ELU
    if normalized == "relu":
        return torch.nn.ReLU
    if normalized == "tanh":
        return torch.nn.Tanh
    if normalized in {"selu", "silu", "swish"}:
        return torch.nn.SiLU if normalized in {"silu", "swish"} else torch.nn.SELU
    raise ValueError(f"Unsupported activation '{name}'. Pass a supported --rsl-activation.")


def _load_agent_activation(checkpoint_path: Path, default: str) -> str:
    agent_yaml = checkpoint_path.parent / "params" / "agent.yaml"
    if not agent_yaml.exists():
        return default
    match = re.search(r"^\s*activation:\s*([A-Za-z0-9_]+)\s*$", agent_yaml.read_text(), re.MULTILINE)
    return match.group(1) if match else default


def _linear_weight_keys(state_dict: dict, prefix: str) -> list[str]:
    weight_keys = [
        key for key in state_dict.keys() if key.startswith(prefix) and key.endswith(".weight")
    ]
    return sorted(weight_keys, key=lambda key: int(key.split(".")[1]))


class TorchModelRunner(ModelRunner):
    """Batched torch model inference."""

    def __init__(self, model, path: Path, device: str, batch_size: int):
        torch = _import_torch()
        self.torch = torch
        self.model = model.to(device).eval()
        self.path = path
        self.device = device
        self.batch_size = batch_size

    def predict(self, obs: np.ndarray) -> np.ndarray:
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, obs.shape[0], self.batch_size):
                batch_np = obs[start : start + self.batch_size].astype(np.float32, copy=False)
                batch = self.torch.from_numpy(batch_np).to(self.device)
                out = self.model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                out = out.detach().cpu().numpy()
                outputs.append(np.asarray(out, dtype=np.float32))
        result = np.concatenate(outputs, axis=0)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result


class OnnxModelRunner(ModelRunner):
    """ONNX Runtime inference, including fixed-batch deployment exports."""

    def __init__(self, path: Path, batch_size: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment check
            raise SystemExit(
                "Missing dependency: onnxruntime. Run inside an environment with "
                "onnxruntime installed when using .onnx models."
            ) from exc

        self.ort = ort
        self.path = path
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.batch_size = batch_size

    def _supports_dynamic_batch(self) -> bool:
        first_dim = self.input_shape[0] if self.input_shape else None
        return first_dim in (None, "batch", "N", -1)

    def _fixed_batch_size(self) -> int | None:
        first_dim = self.input_shape[0] if self.input_shape else None
        return first_dim if isinstance(first_dim, int) and first_dim > 0 else None

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32, copy=False)
        outputs = []
        fixed_batch = self._fixed_batch_size()

        if self._supports_dynamic_batch():
            for start in range(0, obs.shape[0], self.batch_size):
                batch = obs[start : start + self.batch_size]
                outputs.append(self.session.run(None, {self.input_name: batch})[0])
        elif fixed_batch == 1:
            for row in obs:
                outputs.append(self.session.run(None, {self.input_name: row.reshape(1, -1)})[0])
        elif fixed_batch is not None:
            for start in range(0, obs.shape[0], fixed_batch):
                batch = obs[start : start + fixed_batch]
                if batch.shape[0] != fixed_batch:
                    pad = np.repeat(batch[-1:], fixed_batch - batch.shape[0], axis=0)
                    padded = np.concatenate([batch, pad], axis=0)
                    outputs.append(self.session.run(None, {self.input_name: padded})[0][: batch.shape[0]])
                else:
                    outputs.append(self.session.run(None, {self.input_name: batch})[0])
        else:
            raise RuntimeError(f"Unsupported ONNX input shape for {self.path}: {self.input_shape}")

        result = np.concatenate(outputs, axis=0)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result.astype(np.float32, copy=False)


def _build_rsl_actor_from_checkpoint(
    checkpoint_path: Path,
    device: str,
    activation: str,
    batch_size: int,
) -> TorchModelRunner:
    torch = _import_torch()
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    if "actor_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain 'actor_state_dict'.")

    state_dict = checkpoint["actor_state_dict"]
    weight_keys = _linear_weight_keys(state_dict, "mlp.")
    if not weight_keys:
        raise KeyError(f"{checkpoint_path} actor_state_dict does not contain mlp.* weights.")

    activation_cls = _activation_module(activation)
    layers = []
    for layer_index, weight_key in enumerate(weight_keys):
        layer_prefix = ".".join(weight_key.split(".")[:2])
        weight = state_dict[f"{layer_prefix}.weight"]
        bias = state_dict[f"{layer_prefix}.bias"]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if layer_index != len(weight_keys) - 1:
            layers.append(activation_cls())

    model = torch.nn.Sequential(*layers)
    return TorchModelRunner(model=model, path=checkpoint_path, device=device, batch_size=batch_size)


def _build_mlp(input_dim: int, hidden_dims: Iterable[int]):
    torch = _import_torch()
    dims = [input_dim, *list(hidden_dims), 1]
    layers = []
    for idx in range(len(dims) - 2):
        layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(dims[-2], dims[-1]))

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(*layers)

        def forward(self, obs):
            return self.net(obs)

    return MLP()


def _infer_mlp_dims(state_dict: dict) -> tuple[int, tuple[int, ...]]:
    weight_keys = _linear_weight_keys(state_dict, "net.")
    if not weight_keys:
        weight_keys = _linear_weight_keys(state_dict, "")
    if not weight_keys:
        raise KeyError("Could not infer MLP dimensions from checkpoint state dict.")

    weights = [state_dict[key] for key in weight_keys]
    input_dim = int(weights[0].shape[1])
    output_dims = [int(weight.shape[0]) for weight in weights]
    hidden_dims = tuple(output_dims[:-1])
    return input_dim, hidden_dims


def _resolve_safety_value_dir(checkpoint_path: Path, explicit_dir: Path | None) -> Path | None:
    if explicit_dir is not None:
        return explicit_dir.expanduser().resolve()
    if checkpoint_path.parent.name == "models":
        candidate = checkpoint_path.parent.parent
        if (candidate / "training_config.json").exists():
            return candidate.resolve()
    if (checkpoint_path.parent / "training_config.json").exists():
        return checkpoint_path.parent.resolve()
    return None


def _build_safety_value_from_checkpoint(
    checkpoint_path: Path,
    safety_value_dir: Path | None,
    device: str,
    batch_size: int,
) -> TorchModelRunner:
    torch = _import_torch()
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    resolved_dir = _resolve_safety_value_dir(checkpoint_path, safety_value_dir)

    training_config = {}
    if resolved_dir is not None and (resolved_dir / "training_config.json").exists():
        with (resolved_dir / "training_config.json").open("r", encoding="utf-8") as f:
            training_config = json.load(f)

    algorithm = training_config.get("algorithm")
    if "critic1" in checkpoint and "critic2" in checkpoint:
        algorithm = algorithm or "lambda_reachability"
        state_for_dims = checkpoint["critic1"]
    elif "model_state" in checkpoint:
        algorithm = algorithm or "single"
        state_for_dims = checkpoint["model_state"]
    else:
        raise KeyError(
            f"{checkpoint_path} is not a recognized safety-value checkpoint. "
            "Expected 'critic1'/'critic2' or 'model_state'."
        )

    inferred_input_dim, inferred_hidden_dims = _infer_mlp_dims(state_for_dims)
    input_dim = int(training_config.get("input_dim", inferred_input_dim))
    hidden_dims = tuple(int(dim) for dim in training_config.get("hidden_dims", inferred_hidden_dims))

    if algorithm == "lambda_reachability":

        class LambdaValueEnsemble(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.critic1 = _build_mlp(input_dim, hidden_dims)
                self.critic2 = _build_mlp(input_dim, hidden_dims)

            def forward(self, obs):
                return 0.5 * (self.critic1(obs) + self.critic2(obs))

        model = LambdaValueEnsemble()
        model.critic1.load_state_dict(checkpoint["critic1"])
        model.critic2.load_state_dict(checkpoint["critic2"])
    else:
        model = _build_mlp(input_dim, hidden_dims)
        model.load_state_dict(checkpoint["model_state"])

    return TorchModelRunner(model=model, path=checkpoint_path, device=device, batch_size=batch_size)


def load_policy_model(
    model_path: Path,
    device: str = "cpu",
    batch_size: int = 4096,
    rsl_activation: str = "auto",
) -> ModelRunner:
    """Load a deployment policy from ONNX, TorchScript, or raw RSL actor ckpt."""

    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return OnnxModelRunner(model_path, batch_size=batch_size)

    torch = _import_torch()
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        return TorchModelRunner(model=model, path=model_path, device=device, batch_size=batch_size)
    except Exception:
        activation = _load_agent_activation(model_path, "elu") if rsl_activation == "auto" else rsl_activation
        return _build_rsl_actor_from_checkpoint(model_path, device=device, activation=activation, batch_size=batch_size)


def load_safety_value_model(
    model_path: Path,
    safety_value_dir: Path | None = None,
    device: str = "cpu",
    batch_size: int = 4096,
) -> ModelRunner:
    """Load a safety-value model from ONNX, TorchScript, or DPE ckpt."""

    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return OnnxModelRunner(model_path, batch_size=batch_size)

    torch = _import_torch()
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        return TorchModelRunner(model=model, path=model_path, device=device, batch_size=batch_size)
    except Exception:
        return _build_safety_value_from_checkpoint(
            checkpoint_path=model_path,
            safety_value_dir=safety_value_dir,
            device=device,
            batch_size=batch_size,
        )


def _safe_name(rel_name: str) -> str:
    return rel_name.replace("/", "__")


def _metrics(pred: np.ndarray | None, target: np.ndarray) -> dict[str, float]:
    if pred is None:
        return {"mae": math.nan, "rmse": math.nan, "max_abs": math.nan}
    target_view = target.reshape(pred.shape)
    diff = pred - target_view
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _write_summary(summary_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_inference(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    runs = discover_runs(data_root, run_filter=args.run_filter)
    if not runs:
        raise SystemExit(f"No records.csv files found under {data_root}")

    policy_path = resolve_model_arg(
        args.policy_model,
        data_root=data_root,
        metadata_key="policy_model",
        fallbacks=DEPLOY_POLICY_FALLBACKS,
    )
    safety_path = resolve_model_arg(
        args.safety_value_model,
        data_root=data_root,
        metadata_key="safety_value_model",
        fallbacks=DEPLOY_SAFETY_FALLBACKS,
    )

    policy_runner = (
        load_policy_model(policy_path, device=args.device, batch_size=args.batch_size, rsl_activation=args.rsl_activation)
        if policy_path is not None
        else None
    )
    safety_runner = (
        load_safety_value_model(
            safety_path,
            safety_value_dir=args.safety_value_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        if safety_path is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] runs: {len(runs)}")
    print(f"[INFO] policy_model: {policy_path if policy_path else '<disabled>'}")
    print(f"[INFO] safety_value_model: {safety_path if safety_path else '<disabled>'}")

    for run in runs:
        print(f"[INFO] loading {run.rel_name}")
        records = load_real_exp_records(run.records_csv, max_rows=args.max_rows)
        policy_pred = policy_runner.predict(records.obs) if policy_runner else None
        safety_pred = safety_runner.predict(records.obs) if safety_runner else None

        if policy_pred is not None and policy_pred.shape[1] != ACTION_DIM:
            raise RuntimeError(
                f"Policy output for {run.rel_name} has shape {policy_pred.shape}; expected [N, {ACTION_DIM}]"
            )
        if safety_pred is not None:
            safety_pred = safety_pred.reshape(-1, 1)

        policy_stats = _metrics(policy_pred, records.policy_action)
        safety_stats = _metrics(safety_pred, records.safety_value.reshape(-1, 1))

        duration = (
            float(records.relative_time_s[-1] - records.relative_time_s[0])
            if records.relative_time_s.size > 1
            else 0.0
        )
        row = {
            "run": run.rel_name,
            "num_rows": int(records.obs.shape[0]),
            "duration_s": duration,
            "policy_mae": policy_stats["mae"],
            "policy_rmse": policy_stats["rmse"],
            "policy_max_abs": policy_stats["max_abs"],
            "safety_value_mae": safety_stats["mae"],
            "safety_value_rmse": safety_stats["rmse"],
            "safety_value_max_abs": safety_stats["max_abs"],
            "mean_logged_safety_value": float(np.mean(records.safety_value)),
            "min_logged_safety_value": float(np.min(records.safety_value)),
            "max_logged_safety_value": float(np.max(records.safety_value)),
            "mean_safety_signal": float(np.mean(records.safety_signal)),
            "max_safety_signal": float(np.max(records.safety_signal)),
        }
        summary_rows.append(row)

        print(
            "[INFO] "
            f"{run.rel_name}: rows={row['num_rows']} "
            f"policy_mae={row['policy_mae']:.6g} "
            f"safety_value_mae={row['safety_value_mae']:.6g}"
        )

        if args.save_predictions:
            run_output = output_dir / _safe_name(run.rel_name)
            run_output.mkdir(parents=True, exist_ok=True)
            save_payload = {
                "time_s": records.time_s,
                "relative_time_s": records.relative_time_s,
                "control_step": records.control_step,
                "command": records.command,
                "policy_action_logged": records.policy_action,
                "target_q_logged": records.target_q,
                "prev_action_logged": records.prev_action,
                "safety_value_logged": records.safety_value,
                "safety_signal": records.safety_signal,
            }
            if policy_pred is not None:
                save_payload["policy_action_pred"] = policy_pred
            if safety_pred is not None:
                save_payload["safety_value_pred"] = safety_pred.reshape(-1)
            if args.save_obs:
                save_payload["obs"] = records.obs
            np.savez_compressed(run_output / "predictions.npz", **save_payload)

    summary_path = output_dir / "summary.csv"
    _write_summary(summary_path, summary_rows)
    print(f"[INFO] wrote summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load G1 real-exp records.csv files and run policy/safety-value inference."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policy-model",
        type=str,
        default="auto",
        help="Policy model path (.onnx, TorchScript .pt, or raw RSL actor ckpt). Use 'auto' or 'none'.",
    )
    parser.add_argument(
        "--safety-value-model",
        type=str,
        default="auto",
        help="Safety value model path (.onnx, TorchScript .pt, or DPE ckpt). Use 'auto' or 'none'.",
    )
    parser.add_argument(
        "--safety-value-dir",
        type=Path,
        default=None,
        help="Directory containing training_config.json for a DPE safety-value .pt checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for .pt models.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for batched models.")
    parser.add_argument(
        "--rsl-activation",
        type=str,
        default="auto",
        help="Activation for raw RSL actor ckpt. 'auto' reads params/agent.yaml when available.",
    )
    parser.add_argument("--run-filter", type=str, default=None, help="Only process runs whose relative path contains this.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows per run for smoke tests.")
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-predictions",
        dest="save_predictions",
        action="store_true",
        help="Save per-run predictions.npz files.",
    )
    save_group.add_argument(
        "--no-save-predictions",
        dest="save_predictions",
        action="store_false",
        help="Only write summary.csv.",
    )
    parser.set_defaults(save_predictions=True)
    parser.add_argument(
        "--save-obs",
        action="store_true",
        help="Also save obs [N,480] into predictions.npz. This can make outputs much larger.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
