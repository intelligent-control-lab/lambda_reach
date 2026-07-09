#!/usr/bin/env python3
"""Plot training curves across methods for each task.

Reads per-method aggregated training metrics (mean and variance) and produces
comparison plots with ±1σ shading, saved under results/_train/ for each task.
Falls back to mean-only `training_metrics.csv` when variance is unavailable.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Figure sizing and font controls
FIG_WIDTH = 6
FIG_HEIGHT_PER_METRIC = 4
TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 12
SAVE_DPI = 500
LEGEND_VPAD = 0.05  # vertical padding (in figure coords) below last subplot
XLABEL_Y_OFFSET = -0.03  # in axes fraction; moves xlabel to sit at tick level
XLABEL_X_POS = 0.95      # in axes fraction; 1.0 is right end


def _smooth_series(series: pd.Series, window: int = 25) -> pd.Series:
    """Simple moving average with a small window (min_periods=1 for short curves)."""
    window = max(1, int(window))
    return series.rolling(window=window, min_periods=1).mean()


def _load_training_metrics(task: str, method: str) -> Tuple[Optional[pd.DataFrame], Path]:
    """Load per-method training metrics.

    Prefers `training_metrics_aggregated.csv` (mean/var). Falls back to
    `training_metrics.csv` (mean-only) for backward compatibility.
    """
    train_dir = Path("logs/safety_analysis") / task / "results" / method / "train"
    agg_path = train_dir / "training_metrics_aggregated.csv"
    mean_path = train_dir / "training_metrics.csv"

    if agg_path.is_file():
        df = pd.read_csv(agg_path)
        source_path = agg_path
    elif mean_path.is_file():
        df = pd.read_csv(mean_path)
        source_path = mean_path
    else:
        print(f"[WARN] Missing metrics: {agg_path} (and fallback {mean_path})")
        return None, agg_path

    if "step" in df.columns:
        df = df.sort_values("step")
    return df, source_path


def _metric_label(name: str) -> str:
    return name.replace("_", " ").title()


def _to_mathtext(text: str) -> str:
    """Wrap plain text into mathtext using \mathrm; pass through if already mathtext.

    - Replaces spaces with '\\;' and underscores with '\\_'.
    - If the string already contains a '$', assume caller provided mathtext and return as-is.
    """
    if "$" in text:
        return text
    clean = text.replace("\\", "\\\\")
    clean = clean.replace("_", r"\_")
    clean = clean.replace(" ", r"\;")
    return f"$\\mathrm{{{clean}}}$"


def _plot_task(
    task: str,
    methods: Sequence[str],
    display_names: Optional[Sequence[str]] = None,
    metrics_filter: Optional[Sequence[str]] = None,
    label_map: Optional[dict] = None,
    task_label_map: Optional[dict] = None,
    metric_label_map: Optional[dict] = None,
) -> bool:
    """Plot all available metrics for a single task across methods."""
    records: List[Tuple[str, pd.DataFrame]] = []
    for method in methods:
        df, csv_path = _load_training_metrics(task, method)
        if df is None:
            continue
        records.append((method, df))

    # Align step axis to the shortest run so every curve reaches the right edge
    common_max_step: Optional[float] = None
    for _, df in records:
        if "step" not in df.columns or len(df["step"]) == 0:
            continue
        max_step = df["step"].max()
        common_max_step = max_step if common_max_step is None else min(common_max_step, max_step)

    if not records:
        print(f"[WARN] No training metrics found for task {task}; skipping plots")
        return False

    # Determine base metric names, pairing *_mean/_var when available.
    base_metrics: List[str] = []
    if metrics_filter:
        base_metrics = [m for m in metrics_filter if m != "step"]
    else:
        seen = set()
        for _, df in records:
            for col in df.columns:
                if col == "step":
                    continue
                name = col
                if name.endswith("_mean"):
                    name = name[:-5]
                elif name.endswith("_var"):
                    name = name[:-4]
                if name not in seen:
                    seen.add(name)
                    base_metrics.append(name)

    if not base_metrics:
        print(f"[WARN] No metrics to plot for task {task}")
        return False

    out_dir = Path("logs/safety_analysis") / task / "results" / "_train"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_metrics = len(base_metrics)
    fig, axes = plt.subplots(
        num_metrics,
        1,
        figsize=(FIG_WIDTH, FIG_HEIGHT_PER_METRIC * num_metrics),
        squeeze=False,
    )
    axes = axes[:, 0]

    color_cycle = plt.rcParams.get("axes.prop_cycle", plt.cycler(color=[])).by_key().get("color", [])
    legend_handles = []
    legend_labels = []
    any_plotted = False

    display_task = task_label_map.get(task, task) if task_label_map else task

    for m_idx, metric in enumerate(base_metrics):
        ax = axes[m_idx]
        plotted = False
        for idx, (method, df) in enumerate(records):
            mean_col = metric
            var_col = f"{metric}_var"
            if metric not in df.columns and f"{metric}_mean" in df.columns:
                mean_col = f"{metric}_mean"
            if mean_col not in df.columns:
                continue

            df_plot = df
            if common_max_step is not None:
                df_plot = df_plot[df_plot["step"] <= common_max_step]

            label = None
            if label_map and method in label_map:
                label = label_map[method]
            elif display_names and idx < len(display_names):
                label = display_names[idx]
            if label is None:
                label = method.replace("_", " ")

            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            y_mean = df_plot[mean_col]
            y_std = None
            if var_col in df_plot.columns:
                y_std = (df_plot[var_col].clip(lower=0).pow(0.5))

            # Smooth curve for readability; plot raw curve lightly underneath using same color.
            y_mean_smooth = _smooth_series(y_mean, window=25)

            smooth_line = ax.plot(
                df_plot["step"],
                y_mean_smooth,
                label=_to_mathtext(label) if m_idx == 0 else None,
                linewidth=2.0,
                color=color,
            )
            smooth_color = smooth_line[0].get_color()
            ax.plot(
                df_plot["step"],
                y_mean,
                label=None,
                linewidth=1.0,
                alpha=0.25,
                color=smooth_color,
            )

            if y_std is not None:
                ax.fill_between(
                    df_plot["step"],
                    y_mean_smooth - y_std,
                    y_mean_smooth + y_std,
                    alpha=0.2,
                    color=smooth_color,
                )
            if m_idx == 0:
                legend_handles.append(smooth_line[0])
                legend_labels.append(_to_mathtext(label))
            plotted = True
            any_plotted = True

        # Metric labels can differ between y-label and title; nested JSON keys supported.
        metric_y_map = metric_label_map.get("y", {}) if isinstance(metric_label_map, dict) else {}
        metric_title_map = metric_label_map.get("title", {}) if isinstance(metric_label_map, dict) else {}
        metric_y_default = metric_label_map.get("y_default") if isinstance(metric_label_map, dict) else None
        metric_title_default = metric_label_map.get("title_default") if isinstance(metric_label_map, dict) else None

        display_metric_y = metric_y_map.get(metric, metric_y_default or _metric_label(metric))
        display_metric_title = metric_title_map.get(metric, metric_title_default or _metric_label(metric))

        ax.set_xlabel(_to_mathtext("Step"), fontsize=LABEL_FONTSIZE)
        ax.xaxis.set_label_coords(XLABEL_X_POS, XLABEL_Y_OFFSET)
        ax.set_ylabel(_to_mathtext(display_metric_y), fontsize=LABEL_FONTSIZE)
        title_text = f"{display_task} : {display_metric_title}"
        ax.set_title(_to_mathtext(title_text), fontsize=TITLE_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        if common_max_step is not None:
            ax.set_xlim(left=0, right=common_max_step)
        # Clamp value MSE plots to a readable range.
        if "value_mse" in metric:
            ax.set_ylim(0.0, 3.0)

    # Layout first so we can position legend relative to the final axes boxes.
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    if legend_handles:
        last_bbox = axes[-1].get_position()
        legend_y = max(0.01, last_bbox.y0 - LEGEND_VPAD)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=3,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
        )

    if any_plotted:
        out_path = out_dir / f"{task}_train_metrics_column.png"
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"[INFO] Saved all training metrics for {task} to {out_path}")
        plt.close(fig)
        return True

    plt.close(fig)
    print(f"[WARN] No metrics plotted for task {task}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves across methods for each task")
    parser.add_argument("--sa_roots", nargs="+", required=True, help="Safety analysis roots (tasks) to process")
    parser.add_argument("--methods", nargs="+", required=True, help="Method directories (model_subdirs) to include")
    parser.add_argument(
        "--display_names",
        nargs="+",
        help="Optional display names matching --methods (used for legends only)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional list of metrics to plot (defaults to all metrics found in training_metrics.csv)",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=str(Path(__file__).with_name("label_map.json")),
        help="Path to JSON mapping method -> custom label (overrides --display_names); defaults to label_map.json next to this script",
    )
    parser.add_argument(
        "--task_label_map",
        type=str,
        default=str(Path(__file__).with_name("task_label_map.json")),
        help="Path to JSON mapping task -> custom display label (used in plot titles); defaults to task_label_map.json next to this script",
    )
    parser.add_argument(
        "--metric_label_map",
        type=str,
        default=str(Path(__file__).with_name("metric_label_map.json")),
        help="Path to JSON mapping metric -> custom display label (used in y-labels and titles); defaults to metric_label_map.json next to this script",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure display_names aligns with methods
    display_names: Optional[List[str]] = None
    if args.display_names:
        if len(args.display_names) != len(args.methods):
            print(
                f"[WARN] --display_names length ({len(args.display_names)}) does not match --methods ({len(args.methods)}); ignoring custom names"
            )
        else:
            display_names = args.display_names

    # Optional label map overrides display_names; expected JSON: {"method_subdir": "Pretty Label", ...}
    label_map: Optional[dict] = None
    if args.label_map:
        label_path = Path(args.label_map)
        if not label_path.is_file():
            print(f"[WARN] --label_map file not found: {label_path}; ignoring")
        else:
            with open(label_path, "r") as f:
                label_map = json.load(f)

    task_label_map: Optional[dict] = None
    if args.task_label_map:
        task_label_path = Path(args.task_label_map)
        if not task_label_path.is_file():
            print(f"[WARN] --task_label_map file not found: {task_label_path}; ignoring")
        else:
            with open(task_label_path, "r") as f:
                task_label_map = json.load(f)

    metric_label_map: Optional[dict] = None
    if args.metric_label_map:
        metric_label_path = Path(args.metric_label_map)
        if not metric_label_path.is_file():
            print(f"[WARN] --metric_label_map file not found: {metric_label_path}; ignoring")
        else:
            with open(metric_label_path, "r") as f:
                metric_label_map = json.load(f)

    any_plotted = False
    for task in args.sa_roots:
        ok = _plot_task(
            task,
            args.methods,
            display_names=display_names,
            metrics_filter=args.metrics,
            label_map=label_map,
            task_label_map=task_label_map,
            metric_label_map=metric_label_map,
        )
        any_plotted = any_plotted or ok

    if not any_plotted:
        print("[WARN] No plots were generated")


if __name__ == "__main__":
    main()
