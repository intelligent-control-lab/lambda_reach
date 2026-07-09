#!/usr/bin/env python3
"""
Master script to automate training, testing, inference, and evaluation of safety analysis methods.

This script orchestrates the entire pipeline:
1. Train each method on each task
2. Test each trained model on held-out test data
3. Run inference with trained models on simulated environments
4. Evaluate and compare inference results across methods

Usage:
    python safety_value/scripts/run_pipeline.py
"""

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass


# Task configurations: maps safety_analysis_root to metadata about the task and policy checkpoint.
#
# Each config supports two ways of referencing policy checkpoints:
#   1. Provide an explicit "checkpoint" path
#   2. Provide "runner" (experiment folder under logs/rsl_rl) and "checkpoint_step" so the
#      script can automatically pick the latest run folder containing the requested checkpoint.
TASK_CONFIGS = {
    "g1_flat_ppo_1000": {
        "task": "Isaac-Velocity-Flat-G1-PPO",
        "runner": "g1_flat_ppo",
        "checkpoint_step": 1000,
    },
    "g1_rough_ppo_1000": {
        "task": "Isaac-Velocity-Rough-G1-PPO",
        "runner": "g1_rough_ppo",
        "checkpoint_step": 1000,
    },
    "g1_collision_avoid_flat_ppo_1450": {
        "task": "Isaac-Collision-Avoid-Flat-G1-PPO",
        "runner": "g1_collision_avoid_flat_ppo",
        "checkpoint_step": 1450,
    },
    # Add more task configurations here, using either "checkpoint" or ("runner" + "checkpoint_step").
}

DEFAULT_CONFIG_PATH = "safety_value/scripts/config/pipeline_algorithms.json"

# Runtime algorithm spec after expanding ablations
def load_algorithm_specs(config_path: str) -> Tuple[Dict[str, dict], dict]:
    with open(config_path, "r") as f:
        raw = json.load(f)

    defaults = raw.get("defaults", {})
    algos = raw.get("algorithms", {})
    specs: Dict[str, dict] = {}

    for algo_name, cfg in algos.items():
        if not cfg.get("enabled", False):
            continue

        tag = cfg.get("tag", algo_name)
        base_params = cfg.get("train_params", {})
        include_base = cfg.get("include_base", False)  # allow ablation-only runs via config

        # Base entry (optional)
        if include_base:
            specs[algo_name] = {
                "id": algo_name,
                "algo": algo_name,
                "tag": tag,
                "variant": None,
                "train_params": base_params,
            }

        # Ablations (variants share same algo but override params)
        for variant, overrides in cfg.get("ablations", {}).items():
            if isinstance(overrides, dict) and not overrides.get("enabled", False):
                continue

            vid = f"{algo_name}_{variant}"
            merged = base_params.copy()
            merged.update(overrides or {})
            specs[vid] = {
                "id": vid,
                "algo": algo_name,
                "tag": tag,
                "variant": variant,
                "train_params": merged,
            }

    return specs, defaults

# Default inference parameters for Mode 1 (single seed, many envs)
DEFAULT_INFERENCE_PARAMS_MODE1 = {
    "video": True,
    "video_length": 1000,
    "num_envs": 2048,
    "seed": 2345,
}

# Default inference parameters for Mode 2 (multiple seeds, one env each)
DEFAULT_INFERENCE_PARAMS_MODE2 = {
    "video": True,
    "video_length": 1000,
    "num_envs": 1,
    "seeds": [100, 200, 300, 400, 500],  # List of seeds to run
}


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    tasks: List[str]
    algorithms: List[str]  # list of algorithm IDs (may include variants)
    steps: List[str]  # Which steps to run: train, test, plot_train, inference, evaluate
    inference_mode: int  # 1: single seed + many envs, 2: multiple seeds + one env each
    suffix: str  # model subdir suffix; subdir is {algo}_{suffix}
    base_defaults: Dict
    inference_params: Dict
    dry_run: bool
    algo_specs: Dict[str, dict]
    force_rerun: bool


@dataclass
class PlannedStep:
    kind: str
    description: str
    cmd: Optional[List[str]] = None
    run_fn: Optional[Callable[[], bool]] = None


TaskConfig = Dict[str, object]


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def resolve_policy_checkpoint(task_key: str, task_cfg: TaskConfig) -> str:
    """Return an absolute path to the policy checkpoint for a task."""

    checkpoint = task_cfg.get("checkpoint")
    if checkpoint:
        return str(Path(checkpoint).expanduser())

    runner = task_cfg.get("runner")
    step = task_cfg.get("checkpoint_step")
    if not runner or step is None:
        raise ValueError(
            f"Task '{task_key}' must provide either 'checkpoint' or both 'runner' and 'checkpoint_step'."
        )

    checkpoint_root = Path(task_cfg.get("checkpoint_root", "logs/rsl_rl"))
    runner_dir = checkpoint_root / runner
    if not runner_dir.is_dir():
        raise FileNotFoundError(f"Runner directory not found for task '{task_key}': {runner_dir}")

    run_dirs = sorted((d for d in runner_dir.iterdir() if d.is_dir()), reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / f"model_{step}.pt"
        if candidate.is_file():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Could not find model_{step}.pt under runner directory for task '{task_key}': {runner_dir}"
    )


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_step(step_num: int, text: str):
    """Print formatted step."""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}] {text}{Colors.ENDC}")


def print_command(cmd: str):
    """Print command to be executed."""
    print(f"{Colors.OKBLUE}Command: {cmd}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def run_command(cmd: List[str], description: str, dry_run: bool = False) -> bool:
    """Run a shell command and return success status.
    
    Args:
        cmd: Command as list of strings
        description: Description of what the command does
        dry_run: If True, only print command without executing
        
    Returns:
        True if command succeeded, False otherwise
    """
    cmd_str = " ".join(cmd)
    print_command(cmd_str)
    
    if dry_run:
        print_warning("DRY RUN: Command not executed")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print_success(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        print_error(f"Error: {e}")
        return False


def format_cmd(cmd: List[str]) -> str:
    """Format a command list for copy/paste into a shell."""
    return shlex.join(cmd)


def compute_run_name(algo_spec: dict, suffix: str) -> str:
    variant = algo_spec.get("variant")
    base_suffix = "" if suffix == "default" else suffix
    if variant and base_suffix:
        return f"{base_suffix}_{variant}"
    if variant and not base_suffix:
        return variant
    return base_suffix


def get_model_subdir(base_name: str, run_name: str) -> str:
    """Get model subdirectory name using the display tag (or algo if no tag)."""
    if run_name:
        return f"{base_name}_{run_name}"
    return base_name


def merge_params(base_defaults: dict, algo_params: dict) -> dict:
    merged = base_defaults.copy()
    merged.update(algo_params or {})
    return merged


def training_already_done(task: str, algo_spec: dict, run_name: str, seed: int) -> Tuple[bool, Path]:
    """Check if training for a specific seed already produced metrics.

    Returns (done, metrics_path) where metrics_path is where the check was performed.
    """
    model_subdir = get_model_subdir(algo_spec.get("tag", algo_spec["algo"]), run_name)
    metrics_path = Path(
        f"logs/safety_analysis/{task}/results/{model_subdir}/train/seed_{seed}/training_metrics_seed_{seed}.csv"
    )
    return metrics_path.is_file(), metrics_path


def aggregate_training_metrics(task: str, algo_spec: dict, run_name: str, train_seeds: List[int]) -> bool:
    """Aggregate per-seed training metrics into mean/variance CSV and plots.

    Produces:
        - training_metrics_aggregated.csv with *_mean and *_var columns
        - metric_mean_var.png plots with mean curve and ±1 std shading per metric
    """
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print_error(f"Missing plotting dependencies for aggregation: {exc}")
        return False

    model_subdir = get_model_subdir(algo_spec.get("tag", algo_spec["algo"]), run_name)
    train_dir = Path(f"logs/safety_analysis/{task}/results/{model_subdir}/train")
    if not train_dir.exists():
        print_warning(f"Train directory missing, cannot aggregate: {train_dir}")
        return False

    seed_train_frames = []
    seed_eval_frames = []
    for seed in train_seeds:
        seed_dir = train_dir / f"seed_{seed}"
        train_path = seed_dir / f"training_metrics_seed_{seed}.csv"
        eval_path = seed_dir / f"evaluation_metrics_seed_{seed}.csv"

        if train_path.is_file():
            df_train = pd.read_csv(train_path)
            df_train["seed"] = seed
            seed_train_frames.append(df_train)
        else:
            print_warning(f"Missing training metrics for seed {seed}: {train_path}")

        if eval_path.is_file():
            df_eval = pd.read_csv(eval_path)
            df_eval["seed"] = seed
            seed_eval_frames.append(df_eval)
        else:
            print_warning(f"Missing evaluation metrics for seed {seed}: {eval_path}")

    def _aggregate(seed_frames: list, label: str) -> bool:
        if not seed_frames:
            print_warning(f"No {label} metrics found for aggregation in {train_dir}")
            return False

        df_all = pd.concat(seed_frames, ignore_index=True)
        if "step" not in df_all.columns:
            print_warning(f"{label.title()} metrics missing 'step' column in {train_dir}")
            return False

        metrics = [c for c in df_all.columns if c not in {"step", "seed"}]
        grouped = df_all.groupby("step")
        mean_df = grouped[metrics].mean().add_suffix("_mean")
        var_df = grouped[metrics].var(ddof=0).add_suffix("_var")
        agg_df = pd.concat([mean_df, var_df], axis=1).reset_index().sort_values("step")

        agg_path = train_dir / f"{label}_metrics_aggregated.csv"
        agg_df.to_csv(agg_path, index=False)
        print_success(f"Saved aggregated {label} metrics to {agg_path}")

        # Plots per metric
        for base in metrics:
            mean_col = f"{base}_mean"
            var_col = f"{base}_var"
            if mean_col not in agg_df.columns:
                continue
            steps = agg_df["step"].to_numpy()
            mean_vals = agg_df[mean_col].to_numpy()
            std_vals = np.sqrt(agg_df[var_col].fillna(0).to_numpy()) if var_col in agg_df else None

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(steps, mean_vals, label=f"{base} mean", linewidth=2.0)
            if std_vals is not None:
                ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.25, label="±1σ")
            ax.set_xlabel("Step")
            ax.set_ylabel(base)
            ax.set_title(f"{task} | {model_subdir} | {label} {base} (mean±std across seeds)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=9)
            plt.tight_layout()
            # Prefix task to make plots unambiguous across tasks
            out_path = train_dir / f"{task}_{label}_{base}_mean_var.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved aggregated {label} plot to {out_path}")
        return True

    ok_train = _aggregate(seed_train_frames, "training")
    ok_eval = _aggregate(seed_eval_frames, "evaluation")
    return ok_train or ok_eval


def build_train_cmd(task: str, algo_spec: dict, base_defaults: dict, run_name: str, seed: int) -> List[str]:
    algo = algo_spec["algo"]
    params = merge_params(base_defaults, algo_spec.get("train_params", {}))
    epochs = params.get("epochs", 20)
    total_steps = params.get("total_steps")

    cmd: List[str] = [
        "python",
        "safety_value/scripts/2_safety_analysis/train.py",
        "--safety_analysis_root",
        task,
        "--algo",
        algo,
        "--model_tag",
        algo_spec.get("tag", algo),
        "--run_name",
        run_name,
        "--epochs",
        str(epochs),
        "--batch",
        str(params.get("batch", base_defaults.get("batch", 256))),
        "--lr",
        str(params.get("lr", base_defaults.get("lr", 1e-3))),
        "--hidden_dims",
        *[str(d) for d in params.get("hidden_dims", base_defaults.get("hidden_dims", [256, 256]))],
        "--eval_steps",
        str(params.get("eval_steps", base_defaults.get("eval_steps", 1000))),
        "--eval_batches",
        str(params.get("eval_batches", base_defaults.get("eval_batches", 0))),
        "--seed",
        str(seed),
    ]

    if total_steps is not None:
        cmd.extend(["--total_steps", str(total_steps)])

    if algo == "dpe":
        cmd.extend(
            [
                "--lam",
                str(params.get("lam", 0.0)),
                "--lambda_update_steps",
                str(params.get("lambda_update_steps", 500)),
                "--lambda_max",
                str(params.get("lambda_max", 0.99)),
                "--lambda_increase_ratio",
                str(params.get("lambda_increase_ratio", 0.1)),
                "--loss_converge_tol",
                str(params.get("loss_converge_tol", 0.01)),
            ]
        )
        if params.get("use_target_network", True):
            cmd.append("--use_target_network")
            cmd.extend(
                [
                    "--target_update_steps",
                    str(params.get("target_update_steps", 10)),
                    "--target_tau",
                    str(params.get("target_tau", 0.05)),
                ]
            )
        else:
            cmd.append("--no_target_network")
    elif algo == "weakly_supervised":
        cmd.extend(["--margin", str(params.get("margin", 0.1))])
    elif algo == "lambda_reachability":
        cmd.extend(
            [
                "--lambda_param",
                str(params.get("lambda_param", 0.99)),
                "--max_horizon",
                str(params.get("max_horizon", 200)),
                "--alpha_bce",
                str(params.get("alpha_bce", 5.0)),
                "--weight_main",
                str(params.get("weight_main", 1.0)),
                "--weight_hinge_lb",
                str(params.get("weight_hinge_lb", 0.2)),
                "--weight_hinge_mono",
                str(params.get("weight_hinge_mono", 0.2)),
                "--weight_bce",
                str(params.get("weight_bce", 0.2)),
                "--target_tau_lambda",
                str(params.get("target_tau_lambda", 0.05)),
                "--target_update_period",
                str(params.get("target_update_period", 10)),
            ]
        )

    return cmd

def build_test_cmd(task: str, algo_spec: dict, run_name: str) -> List[str]:
    model_subdir = get_model_subdir(algo_spec.get("tag", algo_spec["algo"]), run_name)
    return [
        "python",
        "safety_value/scripts/2_safety_analysis/test.py",
        "--safety_analysis_root",
        task,
        "--model_subdir",
        model_subdir,
    ]


def test_model(task: str, algo_spec: dict, run_name: str, config: PipelineConfig) -> bool:
    """Test a trained model on held-out test data.
    
    Args:
        task: Safety analysis root (task name)
        algo_spec: Algorithm specification (may include variant)
        run_name: Run name with optional variant suffix
        config: Pipeline configuration
        
    Returns:
        True if testing succeeded
    """
    print_step(2, f"Testing {algo_spec['id']} on {task}")

    cmd = build_test_cmd(task, algo_spec, run_name)
    
    return run_command(cmd, f"Testing {algo_spec['id']} on {task}", config.dry_run)


def build_inference_cmd(
    task: str,
    algo_spec: dict,
    seed: int,
    checkpoint_path: str,
    run_name: str,
    config: PipelineConfig,
    task_config: TaskConfig,
) -> List[str]:
    model_subdir = get_model_subdir(algo_spec.get("tag", algo_spec["algo"]), run_name)
    num_envs = config.inference_params.get("num_envs", 2048) if config.inference_mode == 1 else 1

    cmd: List[str] = [
        "python",
        "safety_value/scripts/2_safety_analysis/inference.py",
        "--task",
        str(task_config["task"]),
        "--safety_analysis_root",
        task,
        "--model_subdir",
        model_subdir,
        "--checkpoint",
        checkpoint_path,
        "--seed",
        str(seed),
        "--video_length",
        str(config.inference_params.get("video_length", 1000)),
        "--num_envs",
        str(num_envs),
        "--headless",
    ]
    if config.inference_params.get("video", True):
        cmd.append("--video")
    return cmd


def aggregate_test_results(task: str, algo_ids: List[str], config: PipelineConfig) -> bool:
    """Aggregate test results from all methods for a task.
    
    Args:
        task: Safety analysis root (task name)
        algorithms: List of algorithm names
        config: Pipeline configuration
        
    Returns:
        True if aggregation succeeded
    """
    print_step(2.5, f"Aggregating test results for {task}")
    
    if config.dry_run:
        print_warning("DRY RUN: Skipping aggregation")
        return True
    
    base_dir = Path(f"logs/safety_analysis/{task}")
    eval_dir = base_dir / "results" / "_test_aggregated"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results from all methods
    all_results = {}
    for algo_id in algo_ids:
        spec = config.algo_specs[algo_id]
        run_name = compute_run_name(spec, config.suffix)
        model_subdir = get_model_subdir(spec.get("tag", spec["algo"]), run_name)
        test_dir = base_dir / "results" / model_subdir / "test"
        
        # Find latest results file
        result_files = list(test_dir.glob("results_step_*.json"))
        if not result_files:
            print_warning(f"No test results found for {algo_id}")
            continue
        
        # Get the latest result file (highest step number)
        latest_result = max(result_files, key=lambda p: int(p.stem.split("_")[-1]))
        
        with open(latest_result) as f:
            results = json.load(f)
        
        all_results[algo_id] = {
            "checkpoint_step": results["checkpoint_step"],
            "metrics": results["metrics"],
            "confusion_matrix": results["confusion_matrix"],
        }
    
    # Save aggregated results
    output_file = eval_dir / "aggregated_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print_success(f"Aggregated results saved to {output_file}")
    
    # Print summary table
    print(f"\n{Colors.BOLD}Test Results Summary for {task}:{Colors.ENDC}")
    print(f"{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 68)
    for algo_id, results in all_results.items():
        metrics = results["metrics"]
        print(f"{algo_id:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
    
    return True


def run_inference(task: str, algo_spec: dict, run_name: str, config: PipelineConfig) -> bool:
    """Run inference with a trained model.
    
    Args:
        task: Safety analysis root (task name)
        algo_spec: Algorithm specification
        run_name: Run name with variant suffix
        config: Pipeline configuration
        
    Returns:
        True if inference succeeded
    """
    if task not in TASK_CONFIGS:
        print_error(f"Task {task} not found in TASK_CONFIGS")
        return False
    
    task_config = TASK_CONFIGS[task]
    try:
        checkpoint_path = resolve_policy_checkpoint(task, task_config)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        return False
    model_subdir = get_model_subdir(algo_spec.get("tag", algo_spec["algo"]), run_name)
    
    # Determine seeds and num_envs based on mode
    if config.inference_mode == 1:
        # Mode 1: Single seed, many envs
        seeds = [config.inference_params.get("seed", 2345)]
        num_envs = config.inference_params.get("num_envs", 2048)
    else:
        # Mode 2: Multiple seeds, one env each
        seeds = config.inference_params.get("seeds", [100, 200, 300, 400, 500])
        num_envs = 1
    
    success = True
    for seed in seeds:
        print_step(3, f"Running inference for {algo_spec['id']} on {task} (seed={seed})")

        cmd = build_inference_cmd(task, algo_spec, seed, checkpoint_path, run_name, config, task_config)
        
        if not run_command(cmd, f"Inference {algo_spec['id']} on {task} (seed={seed})", config.dry_run):
            success = False
            if config.inference_mode == 2:
                # In mode 2, ask whether to continue with other seeds
                if not get_yes_no(f"Continue with remaining seeds?", default=True):
                    return False
            else:
                return False
    
    return success


def build_evaluate_cmd(tasks: List[str], methods: List[str], seed: int, config: PipelineConfig) -> List[str]:
    return (
        [
            "python",
            "safety_value/scripts/3_result_analysis/evaluate_inference.py",
            "--sa_roots",
        ]
        + tasks
        + ["--methods"]
        + methods
        + [
            "--seed",
            str(seed),
            "--plot_worst",
            "10",
            "--plot_best",
            "10",
        ]
    )


def build_plot_train_cmd(tasks: List[str], methods: List[str]) -> List[str]:
    return [
        "python",
        "safety_value/scripts/3_result_analysis/plot_training_curves.py",
        "--sa_roots",
        *tasks,
        "--methods",
        *methods,
    ]


def evaluate_inference(tasks: List[str], algo_ids: List[str], config: PipelineConfig) -> bool:
    """Evaluate inference results across methods.
    
    Args:
        tasks: List of safety analysis roots (task names)
        algorithms: List of algorithm names
        config: Pipeline configuration
        
    Returns:
        True if evaluation succeeded
    """
    print_step(4, f"Evaluating inference results")
    
    # Get model subdirs for all algorithms (variants included)
    methods = []
    for algo_id in algo_ids:
        spec = config.algo_specs[algo_id]
        run_name = compute_run_name(spec, config.suffix)
        methods.append(get_model_subdir(spec.get("tag", spec["algo"]), run_name))
    
    # Determine seeds based on mode
    if config.inference_mode == 1:
        # Mode 1: Single seed
        seeds = [config.inference_params.get("seed", 2345)]
    else:
        # Mode 2: Multiple seeds
        seeds = config.inference_params.get("seeds", [100, 200, 300, 400, 500])
    
    success = True
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Evaluating results for seed={seed}")
        print(f"{'='*80}")
        
        cmd = build_evaluate_cmd(tasks, methods, seed, config)
        
        if not run_command(cmd, f"Evaluating inference results (seed={seed})", config.dry_run):
            success = False
            if config.inference_mode == 2:
                # In mode 2, ask whether to continue with other seeds
                if not get_yes_no(f"Continue evaluating remaining seeds?", default=True):
                    return False
            else:
                return False
    
    # If Mode 2, aggregate comparison plots after all seeds are evaluated
    if config.inference_mode == 2 and success:
        for task in tasks:
            if not aggregate_comparison_plots(task, methods, seeds, config):
                print_warning(f"Failed to aggregate comparison plots for {task}")
    
    return success


def prepare_reporting_table(tasks: List[str], algo_ids: List[str], config: PipelineConfig) -> bool:
    """Print a reporting-ready table from aggregated inference CSVs.

    Uses the smallest available seed, reads the aggregated inference CSV produced by
    evaluation, and prints mean±std for selected metrics per method.
    """
    if config.dry_run:
        print_warning("DRY RUN: Skipping reporting table generation")
        return True

    # Reuse the same label map as training plots for consistent display names.
    label_map: Optional[dict] = None
    label_map_path = Path(__file__).parent / "3_result_analysis" / "label_map.json"
    if label_map_path.is_file():
        try:
            with open(label_map_path, "r") as f:
                label_map = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            print_warning(f"Failed to load label map at {label_map_path}: {exc}")
    else:
        print_warning(f"Label map not found at {label_map_path}; using raw method names")

    def _display_method(method: str) -> str:
        if label_map and method in label_map:
            return label_map[method]
        return method

    methods: List[str] = []
    for algo_id in algo_ids:
        spec = config.algo_specs[algo_id]
        run_name = compute_run_name(spec, config.suffix)
        methods.append(get_model_subdir(spec.get("tag", spec["algo"]), run_name))

    seeds = (
        [config.inference_params.get("seed", 2345)]
        if config.inference_mode == 1
        else config.inference_params.get("seeds", [100, 200, 300, 400, 500])
    )
    seed_to_use = min(seeds) if seeds else config.inference_params.get("seed", 2345)

    def _parse_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fmt(mean: Optional[float], std: Optional[float]) -> str:
        if mean is None:
            return "N/A"
        if std is None:
            return f"{mean:.2f}"
        return f"{mean:.2f} ± {std:.2f}"

    def _fmt_percent(mean: Optional[float], std: Optional[float]) -> str:
        if mean is None:
            return "N/A"
        mean_p = mean * 100.0
        std_p = std * 100.0 if std is not None else None
        if std_p is None:
            return f"{mean_p:.2f}"
        return f"{mean_p:.2f} ± {std_p:.2f}"

    success = True
    for task in tasks:
        csv_path = (
            Path(f"logs/safety_analysis/{task}")
            / "results"
            / "_inference_comparison"
            / f"seed_{seed_to_use}"
            / f"inference_evaluation_seed_{seed_to_use}.csv"
        )

        if not csv_path.is_file():
            print_error(f"Missing aggregated inference CSV for task {task}: {csv_path}")
            success = False
            continue

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        print_header(f"REPORT TABLE | {task} | seed {seed_to_use}")
        print(
            f"{'Method':<30} {'Temporal Recall (%) (mean±std)':<32} "
            f"{'Sample Value Error (mean±std)':<36} {'Invariant FPR (%)':<30}"
        )
        print("-" * 130)

        for method in methods:
            row = next((r for r in rows if r.get("method") == method), None)
            if row is None:
                print_warning(f"No data for method {method} in {csv_path}")
                success = False
                continue

            recall_mean = _parse_float(row.get("temporal_recall_mean"))
            recall_std = _parse_float(row.get("temporal_recall_std"))
            sve_mean = _parse_float(row.get("sample_value_error_mean"))
            sve_std = _parse_float(row.get("sample_value_error_std"))
            fpr_mean = _parse_float(row.get("invariant_fpr"))
            fpr_std = None  # FPR reported without std

            display_method = _display_method(method)

            print(
                f"{display_method:<30} {_fmt_percent(recall_mean, recall_std):<32} "
                f"{_fmt(sve_mean, sve_std):<36} {_fmt_percent(fpr_mean, fpr_std):<30}"
            )

    return success


def build_execution_plan(config: PipelineConfig) -> List[PlannedStep]:
    """Build the full execution plan once, for both display and execution."""
    plan: List[PlannedStep] = []

    if "train" in config.steps:
        for task in config.tasks:
            for algo_id in config.algorithms:
                spec = config.algo_specs[algo_id]
                run_name = compute_run_name(spec, config.suffix)
                params = merge_params(config.base_defaults, spec.get("train_params", {}))
                train_seeds = params.get("train_seeds", config.base_defaults.get("train_seeds", [0]))
                if not isinstance(train_seeds, (list, tuple)) or len(train_seeds) == 0:
                    train_seeds = [0]

                for seed in train_seeds:
                    done, metrics_path = training_already_done(task, spec, run_name, seed)
                    if done and not config.force_rerun:
                        plan.append(
                            PlannedStep(
                                kind="train_skip",
                                description=f"[SKIP] Train {algo_id} on {task} (seed={seed}, metrics at {metrics_path})",
                                cmd=None,
                                run_fn=(lambda mp=metrics_path: (print_warning(f"Skipping training; metrics already exist at {mp}"), True)[1]),
                            )
                        )
                        continue

                    plan.append(
                        PlannedStep(
                            kind="train",
                            description=f"Train {algo_id} on {task} (seed={seed})",
                            cmd=build_train_cmd(task, spec, config.base_defaults, run_name, seed),
                        )
                    )

                plan.append(
                    PlannedStep(
                        kind="train_aggregate",
                        description=f"Aggregate training metrics for {algo_id} on {task}",
                        cmd=None,
                        run_fn=(
                            lambda task=task, spec=spec, run_name=run_name, train_seeds=list(train_seeds):
                            aggregate_training_metrics(task, spec, run_name, train_seeds)
                        ),
                    )
                )

    if "test" in config.steps:
        for task in config.tasks:
            for algo_id in config.algorithms:
                spec = config.algo_specs[algo_id]
                run_name = compute_run_name(spec, config.suffix)
                plan.append(
                    PlannedStep(
                        kind="test",
                        description=f"Test {algo_id} on {task}",
                        cmd=build_test_cmd(task, spec, run_name),
                    )
                )
            # Internal aggregation step (not a shell command)
            plan.append(
                PlannedStep(
                    kind="aggregate_test",
                    description=f"Aggregate test results for {task}",
                    cmd=None,
                    run_fn=(lambda task=task: aggregate_test_results(task, config.algorithms, config)),
                )
            )

    if "plot_train" in config.steps:
        methods: List[str] = []
        for algo_id in config.algorithms:
            spec = config.algo_specs[algo_id]
            run_name = compute_run_name(spec, config.suffix)
            base_tag = spec.get("tag", spec["algo"])
            methods.append(get_model_subdir(base_tag, run_name))

        for task in config.tasks:
            plan.append(
                PlannedStep(
                    kind="plot_train",
                    description=f"Plot training curves for {task}",
                    cmd=build_plot_train_cmd([task], methods),
                )
            )

    if "inference" in config.steps:
        for task in config.tasks:
            if task not in TASK_CONFIGS:
                # Mirror runtime behavior: skip tasks not configured
                continue

            task_config = TASK_CONFIGS[task]
            try:
                checkpoint_path = resolve_policy_checkpoint(task, task_config)
            except (FileNotFoundError, ValueError) as exc:
                # Create failing steps with a clear error so display_plan shows what's wrong
                for algo_id in config.algorithms:
                    plan.append(
                        PlannedStep(
                            kind="inference",
                            description=f"Inference {algo_id} on {task} (checkpoint missing)",
                            cmd=None,
                            run_fn=(lambda exc=exc: (print_error(str(exc)) or False)),
                        )
                    )
                continue

            seeds = (
                [config.inference_params.get("seed", 2345)]
                if config.inference_mode == 1
                else config.inference_params.get("seeds", [100, 200, 300, 400, 500])
            )
            for algo_id in config.algorithms:
                spec = config.algo_specs[algo_id]
                run_name = compute_run_name(spec, config.suffix)
                for seed in seeds:
                    plan.append(
                        PlannedStep(
                            kind="inference",
                            description=f"Inference {algo_id} on {task} (seed={seed})",
                            cmd=build_inference_cmd(task, spec, seed, checkpoint_path, run_name, config, task_config),
                        )
                    )

    if "evaluate" in config.steps:
        methods = []
        for algo_id in config.algorithms:
            spec = config.algo_specs[algo_id]
            run_name = compute_run_name(spec, config.suffix)
            methods.append(get_model_subdir(spec.get("tag", spec["algo"]), run_name))
        seeds = (
            [config.inference_params.get("seed", 2345)]
            if config.inference_mode == 1
            else config.inference_params.get("seeds", [100, 200, 300, 400, 500])
        )
        for seed in seeds:
            plan.append(
                PlannedStep(
                    kind="evaluate",
                    description=f"Evaluate inference results (seed={seed})",
                    cmd=build_evaluate_cmd(config.tasks, methods, seed, config),
                )
            )

        if config.inference_mode == 2:
            for task in config.tasks:
                plan.append(
                    PlannedStep(
                        kind="aggregate_comparison",
                        description=f"Aggregate comparison plots for {task}",
                        cmd=None,
                        run_fn=(
                            lambda task=task, methods=methods, seeds=seeds: aggregate_comparison_plots(
                                task, methods, list(seeds), config
                            )
                        ),
                    )
                )

    if "report_table" in config.steps:
        plan.append(
            PlannedStep(
                kind="report_table",
                description="Prepare reporting tables from aggregated inference data",
                cmd=None,
                run_fn=(
                    lambda tasks=list(config.tasks), algo_ids=list(config.algorithms): prepare_reporting_table(
                        tasks, algo_ids, config
                    )
                ),
            )
        )

    return plan


def run_execution_plan(plan: List[PlannedStep], config: PipelineConfig) -> bool:
    """Run a pre-built plan, optionally prompting on failures like the original flow."""
    success = True
    evaluation_success = True

    for step in plan:
        if step.kind == "aggregate_comparison" and config.inference_mode == 2 and not evaluation_success:
            # Match previous behavior: only aggregate if all evaluations succeeded.
            continue

        if step.cmd is not None:
            ok = run_command(step.cmd, step.description, config.dry_run)
        elif step.run_fn is not None:
            if config.dry_run:
                print_warning(f"DRY RUN: Skipping internal step: {step.description}")
                ok = True
            else:
                ok = step.run_fn()
        else:
            ok = True

        if not ok:
            success = False
            if step.kind == "evaluate":
                evaluation_success = False

            if step.kind in {"train", "test", "aggregate_test"}:
                if not get_yes_no("Continue with next task?", default=True):
                    return False
            elif step.kind in {"inference", "evaluate", "plot_train", "report_table"}:
                if config.inference_mode == 2:
                    if not get_yes_no("Continue with remaining seeds?", default=True):
                        return False
                else:
                    return False

    return success


def aggregate_comparison_plots(task: str, methods: List[str], seeds: List[int], config: PipelineConfig) -> bool:
    """Aggregate comparison plots from all seeds into _inference_comparison directory.
    
    In Mode 2, this function collects all comparison plots from individual seed directories
    and copies them to results/_inference_comparison/n_worst/ and results/_inference_comparison/n_best/ with
    filenames like seed_{seed}_{episode_name}_inference_comparison.png
    
    Args:
        task: Safety analysis root (task name)
        methods: List of method names
        seeds: List of seeds used in evaluation
        config: Pipeline configuration
        
    Returns:
        True if aggregation succeeded
    """
    print(f"\n{'='*80}")
    print(f"Aggregating comparison plots for {task} across {len(seeds)} seeds")
    print(f"{'='*80}")
    
    if config.dry_run:
        print_warning("DRY RUN: Skipping plot aggregation")
        return True
    
    base_dir = Path(f"logs/safety_analysis/{task}")
    comparison_dir = base_dir / "results" / "_inference_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Create n_worst and n_best directories
    for subdir in ["n_worst", "n_best"]:
        (comparison_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # For each seed, copy plots and videos to the aggregated location
    import shutil
    copied_plots = 0
    copied_videos = 0
    
    for seed in seeds:
        # Check in seed-specific plot directory
        seed_plot_dir = base_dir / "results" / "_inference_comparison" / f"seed_{seed}" / "plots"
        
        if not seed_plot_dir.exists():
            print_warning(f"No plots found for seed {seed}")
            continue
            
        # Process both n_worst and n_best subdirectories for plots
        for subdir in ["n_worst", "n_best"]:
            subdir_path = seed_plot_dir / subdir
            if subdir_path.exists():
                for plot_file in subdir_path.glob("seed_*_*_inference_comparison.png"):
                    # Plot filename already includes seed info: seed_{seed}_{episode}_inference_comparison.png
                    # Just copy directly
                    dest = comparison_dir / subdir / plot_file.name
                    shutil.copy2(plot_file, dest)
                    copied_plots += 1
        
        # Copy videos from each method
        seed_video_dir = comparison_dir / f"seed_{seed}" / "videos"
        seed_video_dir.mkdir(parents=True, exist_ok=True)
        
        for method in methods:
            method_video_path = base_dir / "results" / method / "inference" / f"seed_{seed}" / "videos" / "inference" / "rl-video-step-0.mp4"
            
            if method_video_path.exists():
                # Destination: results/_inference_comparison/seed_{seed}/videos/{method}_seed_{seed}.mp4
                dest_video = seed_video_dir / f"{method}_seed_{seed}.mp4"
                shutil.copy2(method_video_path, dest_video)
                copied_videos += 1
    
    print_success(f"Aggregated {copied_plots} comparison plots to {comparison_dir}")
    print(f"  Structure: {comparison_dir}/<n_worst|n_best>/seed_*_episode_*_inference_comparison.png")
    print_success(f"Copied {copied_videos} videos to seed-specific directories")
    print(f"  Structure: {comparison_dir}/seed_*/videos/{{method}}_seed_*.mp4")
    
    return True


def get_user_input(prompt: str, options: List[str], allow_all: bool = True) -> List[str]:
    """Get user input with validation.
    
    Args:
        prompt: Prompt to display
        options: List of valid options
        allow_all: Whether to allow 'all' as an option
        
    Returns:
        List of selected options
    """
    print(f"\n{Colors.BOLD}{prompt}{Colors.ENDC}")
    numbered = [f"{idx+1}) {opt}" for idx, opt in enumerate(options)]
    print("Available options:")
    for line in numbered:
        print(f"  {line}")
    if allow_all:
        print("Press Enter for ALL, or choose by number(s)/name(s) (comma-separated)")
    else:
        print("Choose by number(s)/name(s), comma-separated")
    
    while True:
        user_input = input("> ").strip()

        if allow_all and user_input == "":
            return options
        if allow_all and user_input.lower() == 'all':
            return options

        raw_items = [opt.strip() for opt in user_input.split(',') if opt.strip()]
        if not raw_items:
            print_error("Please enter a selection or press Enter for all")
            continue

        # Map numeric entries to options (1-based); allow names too
        selected: List[str] = []
        invalid: List[str] = []
        for item in raw_items:
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(options):
                    selected.append(options[idx])
                else:
                    invalid.append(item)
            else:
                if item in options:
                    selected.append(item)
                else:
                    invalid.append(item)

        if invalid:
            print_error(f"Invalid options: {', '.join(invalid)}")
            continue

        return selected


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user.
    
    Args:
        prompt: Prompt to display
        default: Default value if user just presses enter
        
    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print_error("Please enter 'y' or 'n'")


def display_plan(config: PipelineConfig):
    """Display execution plan and get confirmation.
    
    Args:
        config: Pipeline configuration
    """
    print_header("EXECUTION PLAN")
    
    print(f"{Colors.BOLD}Tasks:{Colors.ENDC}")
    for task in config.tasks:
        if task in TASK_CONFIGS:
            task_info = TASK_CONFIGS[task]
            print(f"  - {task}")
            print(f"    Task: {task_info['task']}")
            try:
                checkpoint_preview = resolve_policy_checkpoint(task, task_info)
                print(f"    Checkpoint: {checkpoint_preview}")
            except (FileNotFoundError, ValueError) as exc:
                print(f"    Checkpoint: {Colors.FAIL}{exc}{Colors.ENDC}")
        else:
            print(f"  - {task} {Colors.WARNING}(WARNING: Not in TASK_CONFIGS){Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Algorithms:{Colors.ENDC}")
    for algo_id in config.algorithms:
        spec = config.algo_specs[algo_id]
        label = f"{algo_id} (base: {spec['algo']})" if spec.get("variant") else algo_id
        print(f"  - {label}")
    
    print(f"\n{Colors.BOLD}Steps to Execute:{Colors.ENDC}")
    for step in config.steps:
        print(f"  - {step}")
    
    print(f"\n{Colors.BOLD}Parameters:{Colors.ENDC}")
    print(f"  Inference Mode: {config.inference_mode} ({'Single seed, many envs' if config.inference_mode == 1 else 'Multiple seeds, one env each'})")
    if config.inference_mode == 1:
        print(f"    Seed: {config.inference_params.get('seed', 2345)}")
        print(f"    Num Envs: {config.inference_params.get('num_envs', 2048)}")
    else:
        seeds = config.inference_params.get('seeds', [100, 200, 300, 400, 500])
        print(f"    Seeds: {seeds}")
        print(f"    Num Envs per seed: 1")
    print(f"  Training defaults: {config.base_defaults}")
    print(f"  Force rerun existing results: {config.force_rerun}")
    
    print(f"\n{Colors.BOLD}Total Operations:{Colors.ENDC}")
    n_train = 0
    if 'train' in config.steps:
        for algo_id in config.algorithms:
            spec = config.algo_specs[algo_id]
            params = merge_params(config.base_defaults, spec.get("train_params", {}))
            train_seeds = params.get("train_seeds", config.base_defaults.get("train_seeds", [0]))
            if not isinstance(train_seeds, (list, tuple)) or len(train_seeds) == 0:
                train_seeds = [0]
            n_train += len(config.tasks) * len(train_seeds)
    n_test = len(config.tasks) * len(config.algorithms) if 'test' in config.steps else 0
    
    # Calculate inference runs based on mode
    if 'inference' in config.steps:
        if config.inference_mode == 1:
            n_inference = len(config.tasks) * len(config.algorithms)
        else:
            n_seeds = len(config.inference_params.get('seeds', [100, 200, 300, 400, 500]))
            n_inference = len(config.tasks) * len(config.algorithms) * n_seeds
    else:
        n_inference = 0
    
    # Calculate evaluation runs based on mode
    if 'evaluate' in config.steps:
        if config.inference_mode == 1:
            n_evaluate = 1
        else:
            n_evaluate = len(config.inference_params.get('seeds', [100, 200, 300, 400, 500]))
    else:
        n_evaluate = 0

    n_plot_train = len(config.tasks) if 'plot_train' in config.steps else 0
    n_report_table = 1 if 'report_table' in config.steps else 0
    
    print(f"  Training runs: {n_train}")
    print(f"  Testing runs: {n_test}")
    print(f"  Inference runs: {n_inference}")
    print(f"  Evaluation runs: {n_evaluate}")
    print(f"  Training plots: {n_plot_train}")
    print(f"  Reporting tables: {n_report_table}")
    print(f"  Total: {n_train + n_test + n_inference + n_evaluate + n_plot_train + n_report_table}")
    
    print("\n" + "="*80 + "\n")
    
    # Display all commands that will be run
    print(f"{Colors.BOLD}Commands to be executed:{Colors.ENDC}\n")

    plan = build_execution_plan(config)
    for idx, step in enumerate(plan, start=1):
        print(f"{Colors.OKCYAN}[{idx}] {step.description}:{Colors.ENDC}")
        if step.cmd is not None:
            print(f"    {format_cmd(step.cmd)}")
        else:
            print("    (Internal step)")
    
    print("\n" + "="*80 + "\n")


def run_pipeline(config: PipelineConfig):
    """Execute the full pipeline.
    
    Args:
        config: Pipeline configuration
    """
    print_header("STARTING PIPELINE EXECUTION")

    plan = build_execution_plan(config)
    ok = run_execution_plan(plan, config)

    # Final summary
    print_header("PIPELINE EXECUTION COMPLETE")
    if ok:
        print_success("All operations completed successfully!")
    else:
        print_warning("One or more operations failed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Master pipeline for safety analysis training, testing, and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="default",
        help="Model subdir suffix; results are stored under results/<algo>_<suffix>"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline algorithm configuration JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run all steps even if prior results/metrics exist"
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        help="Safety analysis roots to process (overrides interactive input)"
    )
    parser.add_argument(
        "--algorithms",
        nargs='+',
        help="Algorithms to use (overrides interactive input)"
    )
    parser.add_argument(
        "--steps",
        nargs='+',
        choices=['train', 'test', 'plot_train', 'inference', 'evaluate', 'report_table'],
        help="Steps to execute (overrides interactive input)"
    )
    parser.add_argument(
        "--inference_mode",
        type=int,
        choices=[1, 2],
        help="Inference mode: 1=single seed + many envs, 2=multiple seeds + one env each (overrides interactive input)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2345,
        help="Random seed for inference (Mode 1 only)"
    )
    parser.add_argument(
        "--seeds",
        nargs='+',
        type=int,
        help="List of seeds for inference (Mode 2 only, e.g., --seeds 100 200 300)"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        help="Number of random seeds to generate (Mode 2 only, overrides interactive input)"
    )
    
    args = parser.parse_args()
    
    print_header("SAFETY ANALYSIS PIPELINE")

    # Load algorithm specs/config
    try:
        algo_specs, base_defaults = load_algorithm_specs(args.config)
    except FileNotFoundError:
        print_error(f"Config file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print_error(f"Failed to parse config file {args.config}: {exc}")
        sys.exit(1)

    def _algo_priority(algo_id: str) -> int:
        spec = algo_specs.get(algo_id, {})
        base = spec.get("algo", algo_id)
        if base == "dpe":
            return 0
        if base == "lambda_reachability":
            return 1
        return 2

    # Order algorithms: DPE first, then λ-reachability, then all others
    available_algo_ids = sorted(algo_specs.keys(), key=_algo_priority)

    # Get tasks
    if args.tasks:
        tasks = args.tasks
    else:
        available_tasks = list(TASK_CONFIGS.keys())
        if not available_tasks:
            print_error("No tasks configured in TASK_CONFIGS!")
            print("Please edit the script and add task configurations.")
            sys.exit(1)
        tasks = get_user_input("Select tasks to process:", available_tasks, allow_all=True)

    if not tasks:
        print_error("No tasks selected!")
        sys.exit(1)

    # Get algorithms (IDs, includes variants)
    if args.algorithms:
        algorithms = args.algorithms
    else:
        algorithms = get_user_input("Select algorithms to use:", available_algo_ids, allow_all=True)

    if not algorithms:
        print_error("No algorithms selected!")
        sys.exit(1)

    invalid_algos = [a for a in algorithms if a not in available_algo_ids]
    if invalid_algos:
        print_error(f"Unknown algorithms: {invalid_algos}. Available: {available_algo_ids}")
        sys.exit(1)
    
    # Enforce processing order regardless of input sequence
    algorithms = sorted(algorithms, key=_algo_priority)

    # Get steps
    if args.steps:
        steps = args.steps
    else:
        all_steps = ['train', 'test', 'plot_train', 'inference', 'evaluate', 'report_table']
        steps = get_user_input("Select steps to execute:", all_steps, allow_all=True)
    
    if not steps:
        print_error("No steps selected!")
        sys.exit(1)
    
    # Get inference mode (only if inference or evaluate steps are selected)
    if 'inference' in steps or 'evaluate' in steps:
        if args.inference_mode:
            inference_mode = args.inference_mode
        else:
            print(f"\n{Colors.BOLD}Select inference/evaluation mode:{Colors.ENDC}")
            print("  1: Single seed with many environments (fast, less variance)")
            print("  2: Multiple seeds with one environment each (slower, more variance)")
            while True:
                mode_input = input("Enter mode (1 or 2) [default: 1]: ").strip()
                if not mode_input:
                    inference_mode = 1
                    break
                if mode_input in ['1', '2']:
                    inference_mode = int(mode_input)
                    break
                print_error("Please enter 1 or 2")
        
        # Setup inference parameters based on mode
        if inference_mode == 1:
            # Mode 1: Single seed, many envs
            inference_params = DEFAULT_INFERENCE_PARAMS_MODE1.copy()
            if args.seed != 2345:  # User specified a custom seed
                inference_params['seed'] = args.seed
        else:
            # Mode 2: Multiple seeds, one env each
            inference_params = DEFAULT_INFERENCE_PARAMS_MODE2.copy()
            
            # Determine seeds for Mode 2
            if args.seeds:
                # User specified explicit seeds via command line
                seeds_to_use = args.seeds
                print(f"\nUsing {len(seeds_to_use)} seeds from command line: {seeds_to_use}")
            else:
                # Ask user for number of seeds and generate them randomly
                if args.num_seeds:
                    num_seeds = args.num_seeds
                else:
                    print(f"\n{Colors.BOLD}How many random seeds should be generated?{Colors.ENDC}")
                    while True:
                        num_input = input("Enter number of seeds [default: 5]: ").strip()
                        if not num_input:
                            num_seeds = 5
                            break
                        try:
                            num_seeds = int(num_input)
                            if num_seeds > 0:
                                break
                            else:
                                print_error("Please enter a positive number")
                        except ValueError:
                            print_error("Please enter a valid number")
                
                # Generate random seeds
                random.seed()  # Use system time for randomness
                seeds_to_use = [random.randint(1000, 9999) for _ in range(num_seeds)]
                print(f"\nGenerated {num_seeds} random seeds: {seeds_to_use}")
            
            inference_params['seeds'] = seeds_to_use
    else:
        # No inference/evaluation, use default mode 1 params (won't be used)
        inference_mode = 1
        inference_params = DEFAULT_INFERENCE_PARAMS_MODE1.copy()
    
    # Create configuration
    config = PipelineConfig(
        tasks=tasks,
        algorithms=algorithms,
        steps=steps,
        inference_mode=inference_mode,
        suffix=args.suffix,
        base_defaults=base_defaults,
        inference_params=inference_params,
        dry_run=args.dry_run,
        algo_specs=algo_specs,
        force_rerun=args.force_rerun,
    )
    
    # Display plan and get confirmation
    display_plan(config)
    
    if not args.dry_run:
        if not get_yes_no("Proceed with execution?", default=False):
            print("Execution cancelled.")
            sys.exit(0)
    
    # Run pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()
