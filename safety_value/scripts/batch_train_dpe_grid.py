#!/usr/bin/env python3
"""Batch grid training script for DPE.

Runs a sweep of DPE hyperparameters by invoking:
  safety_value/scripts/2_safety_analysis/train.py

After all runs, aggregates the final train/test accuracy from each run's
`train/training_accuracy.csv` and writes a summary CSV under:
  logs/safety_analysis/<safety_analysis_root>/results/

Example:
  python safety_value/scripts/batch_train_dpe_grid.py --safety_analysis_root g1_rough_ppo_1000
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class DpeRunSpec:
    safety_analysis_root: str
    algo: str
    lr: float
    batch_size: int
    hidden_dims: tuple
    epochs: int
    lam: float
    lambda_update_steps: int
    lambda_max: float
    lambda_increase_ratio: float
    loss_converge_tol: float
    use_target_network: bool
    target_update_steps: Optional[int]
    target_tau: Optional[float]
    run_name: str
    results_subdir: str


def _f(x: float) -> str:
    """Float -> compact string safe for filenames."""
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def _make_run_name(
    lr: float,
    batch_size: int,
    hidden_dims: tuple,
    lam: float,
    lambda_update_steps: int,
    lambda_max: float,
    lambda_increase_ratio: float,
    loss_converge_tol: float,
    use_target_network: bool,
    target_update_steps: Optional[int],
    target_tau: Optional[float],
) -> str:
    hidden_str = "x".join(str(d) for d in hidden_dims)
    parts = [
        f"lr{_f(lr)}",
        f"bs{batch_size}",
        f"hd{hidden_str}",
        f"lam{_f(lam)}",
        f"lus{lambda_update_steps}",
        f"lmax{_f(lambda_max)}",
        f"lir{_f(lambda_increase_ratio)}",
        f"lct{_f(loss_converge_tol)}",
        f"tn{1 if use_target_network else 0}",
    ]
    if use_target_network:
        parts.append(f"tus{target_update_steps}")
        parts.append(f"tau{_f(float(target_tau))}")
    return "_".join(parts)


def iter_specs(args: argparse.Namespace) -> Iterable[DpeRunSpec]:
    # Focused local sweep around best-performing hyperparameters
    lr_choices                    = [5e-4]
    batch_size_choices            = [256,]
    hidden_dims_choices           = [(256, 256), (512, 256)]
    lam_choices                   = [0.0,]
    lambda_max_choices            = [0.99, 0.999]
    lambda_update_steps_choices   = [500,]
    lambda_increase_ratio_choices = [0.1,]
    loss_converge_tol_choices     = [0.01,]
    target_tau_choices            = [0.05,]
    target_update_steps_choices   = [10,]
    results_subdir = f"dpe_batch_{getattr(args, 'batch_id', 1)}"

    for (lr, bs, hd, lam, lus, lmax, lir, lct, tus, tau) in itertools.product(
        lr_choices,
        batch_size_choices,
        hidden_dims_choices,
        lam_choices,
        lambda_update_steps_choices,
        lambda_max_choices,
        lambda_increase_ratio_choices,
        loss_converge_tol_choices,
        target_update_steps_choices,
        target_tau_choices,
    ):
        run_name = _make_run_name(
            lr=lr,
            batch_size=bs,
            hidden_dims=hd,
            lam=lam,
            lambda_update_steps=lus,
            lambda_max=lmax,
            lambda_increase_ratio=lir,
            loss_converge_tol=lct,
            use_target_network=True,
            target_update_steps=tus,
            target_tau=tau,
        )
        yield DpeRunSpec(
            safety_analysis_root=args.safety_analysis_root,
            algo="dpe",
            lr=lr,
            batch_size=bs,
            hidden_dims=hd,
            epochs=args.epochs,
            lam=lam,
            lambda_update_steps=lus,
            lambda_max=lmax,
            lambda_increase_ratio=lir,
            loss_converge_tol=lct,
            use_target_network=True,
            target_update_steps=tus,
            target_tau=tau,
            run_name=run_name,
            results_subdir=results_subdir,
        )


def _repo_root() -> Path:
    # This file lives at: <repo>/safety_value/scripts/batch_train_dpe_grid.py
    return Path(__file__).resolve().parents[2]


def _is_run_completed(output_dir: Path) -> bool:
    """Check if a training run is already completed.
    
    A run is considered complete if:
    1. last.pt model checkpoint exists
    2. training_accuracy.csv exists and has data
    """
    last_pt = output_dir / "models" / "last.pt"
    training_csv = output_dir / "train" / "training_accuracy.csv"
    
    if not last_pt.exists():
        return False
    if not training_csv.exists():
        return False
    
    # Check that CSV has content
    try:
        with training_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return len(rows) > 0
    except Exception:
        return False


def _run_training(spec: DpeRunSpec) -> tuple[bool, list[str]]:
    repo_root = _repo_root()
    train_py = repo_root / "dpe" / "scripts" / "2_safety_analysis" / "train.py"

    cmd = [
        sys.executable,
        str(train_py),
        "--safety_analysis_root",
        spec.safety_analysis_root,
        "--algo",
        spec.algo,
        "--lr",
        str(spec.lr),
        "--batch",
        str(spec.batch_size),
        "--hidden_dims",
        *[str(d) for d in spec.hidden_dims],
        "--epochs",
        str(spec.epochs),
        "--lam",
        str(spec.lam),
        "--lambda_update_steps",
        str(spec.lambda_update_steps),
        "--lambda_max",
        str(spec.lambda_max),
        "--lambda_increase_ratio",
        str(spec.lambda_increase_ratio),
        "--loss_converge_tol",
        str(spec.loss_converge_tol),
        "--run_name",
        spec.run_name,
        "--results_subdir",
        spec.results_subdir,
    ]

    if spec.use_target_network:
        cmd.append("--use_target_network")
        cmd.extend(["--target_update_steps", str(spec.target_update_steps)])
        cmd.extend(["--target_tau", str(spec.target_tau)])

    # Suppress output: redirect stdout/stderr to DEVNULL
    ok = subprocess.run(
        cmd,
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    return ok, cmd


def _read_last_accuracy_row(training_accuracy_csv: Path) -> dict:
    """Read last row of train/training_accuracy.csv.

    Expected columns: step, train_acc, test_acc, lambda
    """
    with training_accuracy_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {training_accuracy_csv}")

    last = rows[-1]

    def _to_float(v: str) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    return {
        "final_step": int(float(last.get("step", "nan"))),
        "final_train_acc": _to_float(last.get("train_acc", "nan")),
        "final_test_acc": _to_float(last.get("test_acc", "nan")),
        "final_lambda": _to_float(last.get("lambda", "nan")),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch train DPE over a fixed hyperparameter grid and aggregate results."
    )
    p.add_argument("--safety_analysis_root", required=True, type=str)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--loss_converge_tol", type=float, default=0.01)
    p.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum concurrent train.py processes (default: 8).",
    )
    p.add_argument(
        "--batch_id",
        type=int,
        default=1,
        help="Batch identifier used to group outputs under results/dpe_batch_<id>.",
    )
    p.add_argument(
        "--summary_only",
        action="store_true",
        help="Only generate summary from existing runs without training anything.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    specs = list(iter_specs(args))
    seen = set()
    for s in specs:
        if s.run_name in seen:
            raise RuntimeError(f"Duplicate run_name generated: {s.run_name}")
        seen.add(s.run_name)

    if not specs:
        print("No run specs generated; exiting.")
        return 0

    if args.max_workers < 1:
        raise ValueError("--max_workers must be >= 1")

    repo_root = _repo_root()
    results_root = (
        repo_root / "logs" / "safety_analysis" / args.safety_analysis_root / "results"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    batch_dir = results_root / f"dpe_batch_{args.batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already completed runs
    specs_to_run = []
    skipped_specs = []
    
    if args.summary_only:
        print("Summary-only mode: checking for existing results...")
    else:
        print("Checking for already completed runs...")
    
    for spec in specs:
        output_dir = batch_dir / f"{spec.algo}_{spec.run_name}"
        if _is_run_completed(output_dir):
            skipped_specs.append(spec)
            if not args.summary_only:
                print(f"  ✓ Already finished: {spec.run_name}")
        else:
            if not args.summary_only:
                specs_to_run.append(spec)
    
    total_runs = len(specs)
    runs_to_execute = len(specs_to_run)
    runs_skipped = len(skipped_specs)
    
    if args.summary_only:
        print(f"\nFound {runs_skipped} completed run(s) to summarize")
    else:
        print(f"\nTotal configurations: {total_runs}")
        print(f"Already completed: {runs_skipped}")
        print(f"To execute: {runs_to_execute}")
        
        if runs_skipped > 0:
            print(f"\nSkipped {runs_skipped} completed run(s)")

    summary_rows: list[dict] = []

    if args.summary_only:
        print("\nSkipping training (summary-only mode)")
    elif runs_to_execute == 0:
        print("\nAll runs already completed. Nothing to execute.")
    else:
        print(f"\nStarting execution with {args.max_workers} workers...\n")

    max_workers = min(args.max_workers, runs_to_execute) if runs_to_execute > 0 else 1
    
    if not args.summary_only and runs_to_execute > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_spec = {}
            for spec in specs_to_run:
                future = executor.submit(_run_training, spec)
                future_to_spec[future] = spec

            completed = 0
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                completed += 1
                try:
                    ok, cmd = future.result()
                    err_msg = ""
                except Exception as exc:  # pragma: no cover - defensive
                    ok = False
                    cmd = []
                    err_msg = f"exception: {exc}"

                output_dir = batch_dir / f"{spec.algo}_{spec.run_name}"
                training_csv = output_dir / "train" / "training_accuracy.csv"

                metrics = {}
                status = "ok" if ok else "failed"
                err_msg_local = err_msg
                if ok:
                    try:
                        metrics = _read_last_accuracy_row(training_csv)
                    except Exception as e:
                        status = "missing_metrics"
                        err_msg_local = str(e)
                else:
                    if not err_msg_local:
                        err_msg_local = "train.py returned non-zero"

                row = {
                    "run_name": spec.run_name,
                    "output_dir": str(output_dir),
                    "status": status,
                    "error": err_msg_local,
                    "cmd": " ".join(cmd),
                    **{
                        k: v
                        for k, v in asdict(spec).items()
                        if k not in {"safety_analysis_root", "algo", "run_name", "results_subdir"}
                    },
                    **metrics,
                }
                summary_rows.append(row)
                print(
                    f"[{completed}/{runs_to_execute}] {spec.run_name}: {status}"
                    + (f" - {err_msg_local}" if err_msg_local else "")
                )

    # Add skipped runs to summary with their existing metrics
    for spec in skipped_specs:
        output_dir = batch_dir / f"{spec.algo}_{spec.run_name}"
        training_csv = output_dir / "train" / "training_accuracy.csv"
        
        metrics = {}
        status = "skipped"
        try:
            metrics = _read_last_accuracy_row(training_csv)
        except Exception as e:
            status = "skipped_no_metrics"
        
        row = {
            "run_name": spec.run_name,
            "output_dir": str(output_dir),
            "status": status,
            "error": "",
            "cmd": "",
            **{
                k: v
                for k, v in asdict(spec).items()
                if k not in {"safety_analysis_root", "algo", "run_name", "results_subdir"}
            },
            **metrics,
        }
        summary_rows.append(row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = batch_dir / f"dpe_grid_summary_batch{args.batch_id}_{ts}.csv"

    fieldnames = [
        "run_name",
        "status",
        "error",
        "output_dir",
        "lr",
        "batch_size",
        "hidden_dims",
        "lam",
        "lambda_update_steps",
        "lambda_max",
        "lambda_increase_ratio",
        "loss_converge_tol",
        "use_target_network",
        "target_update_steps",
        "target_tau",
        "final_step",
        "final_train_acc",
        "final_test_acc",
        "final_lambda",
        "cmd",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total configurations: {total_runs}")
    print(f"  Skipped (already done): {runs_skipped}")
    print(f"  Newly executed: {runs_to_execute}")
    print(f"  Results written to: {summary_path}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
