"""Summarize fixed-budget, multiple-seed SPARK value results and paper targets."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    "temporal_recall",
    "sample_value_error",
    "paper_false_safe_rate",
    "false_alarm_rate",
    "invariant_accuracy",
]
PAPER = {
    "lambda_reachability": [0.994, 0.13, 0.0612],
    "supervised": [0.9972, 0.14, 0.0205],
    "dpe": [0.2261, 0.18, 0.6047],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.evaluation / "results.json").read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    if not results["complete"] or not manifest["complete"]:
        raise ValueError("Incomplete experiment")
    summary = {}
    configs = {}
    for method in PAPER:
        rows = []
        for name, result in results["models"].items():
            if name in ("always_safe", "always_unsafe"):
                continue
            config = json.loads((Path(name) / "training_config.json").read_text())
            if config["algorithm"] == method:
                rows.append(result)
                configs[name] = config
        if not rows:
            raise ValueError(f"Missing method {method}")
        summary[method] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "seed_std": float(np.std([r[metric] for r in rows], ddof=1))
                if len(rows) > 1
                else None,
            }
            for metric in METRICS
        }
        summary[method]["training_seeds"] = len(rows)
    paper_source = "https://arxiv.org/html/2606.16022#S4.T1"
    (args.evaluation / "summary.json").write_text(
        json.dumps(
            {
                "methods": summary,
                "paper": PAPER,
                "paper_source": paper_source,
                "paper_comparison": "different benchmark; reference targets only",
            },
            indent=2,
        )
        + "\n"
    )
    text = [
        "# SPARK built-in G1 avoidance value accuracy",
        "",
        "This is a 29-DoF SPARK evaluation using the paper's metric implementation.",
        "The paper's 37-DoF simulation results are reference targets, not a matched population.",
        "",
        "| Method | Temporal recall (%) | Value MSE | False-safe (%) | False-alarm (%) | Sign accuracy (%) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method, stats in summary.items():
        cells = []
        for metric in METRICS:
            scale = 1 if metric == "sample_value_error" else 100
            item = stats[metric]
            cells.append(
                f"{item['mean'] * scale:.3f} ± {(item['seed_std'] or 0) * scale:.3f}"
            )
        text.append("| " + " | ".join([method, *cells]) + " |")
    text += [
        "",
        "Values above are mean ± sample standard deviation across training seeds.",
        "Per-model 95% episode-bootstrap intervals are in `results.json`.",
        "",
        "Paper G1Collision λ reference: temporal recall 99.40%, value MSE 0.13, false-safe 6.12%.",
        f"Source: [paper Table 1]({paper_source}).",
        "",
        "False-safe = predicts safe among observed-future unsafe frames. False-alarm = predicts unsafe among observed-future safe frames.",
        "The published FPR uses the first definition. Value MSE averages segment MSEs as in the supplied evaluation script; frame-weighted MSE is also saved.",
        "",
    ]
    for split, items in manifest["splits"].items():
        text.append(
            f"- {split}: {len(items)} episodes, {sum(r['frames'] for r in items):,} frames, "
            f"{sum(r['unsafe_frames'] for r in items):,} instantaneous unsafe frames, {sum(r['terminated'] for r in items)} terminations."
        )
    example = next(
        r
        for k, r in results["models"].items()
        if k not in ("always_safe", "always_unsafe")
    )
    text += [
        "",
        (
            f"Held-out evaluation: {example['segments']} event segments; {example['unsafe_segments']} unsafe segments; "
            f"{example['unsafe_frames']} observed-future unsafe frames."
        ),
        "",
        "## Controls",
        "",
    ]
    for name in ("always_safe", "always_unsafe"):
        r = results["models"][name]
        text.append(
            f"- {name}: sign accuracy {100 * r['invariant_accuracy']:.2f}%, temporal recall {100 * r['temporal_recall']:.2f}%, "
            f"false-safe {100 * r['paper_false_safe_rate']:.2f}%, false-alarm {100 * r['false_alarm_rate']:.2f}%."
        )
    text += [
        "",
        "## Training",
        "",
        "All model paths and checkpoint hashes are in `results.json`. Test metrics did not select checkpoints.",
        "",
    ]
    for name, cfg in configs.items():
        text.append(
            f"- `{name}`: {cfg['total_steps']} updates, seed {cfg['seed']}, {cfg['balanced_training_samples']} balanced anchors."
        )
    text += [
        "",
        "This evaluates sampled commands and ball disturbances in the nominal built-in scene. It does not qualify hardware, noisy sensors, rough terrain, or push-task values.",
        "A value checkpoint is not promoted to the live viewer automatically.",
        "",
    ]
    (args.evaluation / "REPORT.md").write_text("\n".join(text))
    methods = list(PAPER)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = ["Temporal recall (%) ↑", "Value MSE ↓", "False-safe rate (%) ↓"]
    for i, ax in enumerate(axes):
        metric = METRICS[i]
        scale = 1 if i == 1 else 100
        x = np.arange(3)
        ax.bar(
            x - 0.18,
            [PAPER[m][i] * scale for m in methods],
            0.36,
            label="Paper: 37-DoF Isaac",
            color="#9aa6b2",
        )
        ax.bar(
            x + 0.18,
            [summary[m][metric]["mean"] * scale for m in methods],
            0.36,
            yerr=[(summary[m][metric]["seed_std"] or 0) * scale for m in methods],
            capsize=3,
            label="SPARK: 29-DoF MuJoCo",
            color="#287f8e",
        )
        ax.set_xticks(x, ["λ-reachability", "Supervised", "DPE"])
        ax.set_ylabel(labels[i])
        ax.grid(axis="y", alpha=0.2)
        if i != 1:
            ax.set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Different benchmarks; bars are reference comparisons, not reproduction proof"
    )
    fig.tight_layout()
    fig.savefig(args.evaluation / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in PAPER:
        curves = []
        for name, cfg in configs.items():
            if cfg["algorithm"] != method:
                continue
            path = Path(name) / "train" / f"evaluation_metrics_seed_{cfg['seed']}.csv"
            with path.open() as stream:
                curve = list(csv.DictReader(stream))
            steps = np.array([int(row["step"]) for row in curve])
            curves.append([float(row["eval_value_mse"]) for row in curve])
        mean, std = np.mean(curves, axis=0), np.std(curves, axis=0)
        (line,) = ax.plot(steps, mean, marker="o", label=method)
        ax.fill_between(
            steps,
            np.maximum(1e-6, mean - std),
            mean + std,
            alpha=0.15,
            color=line.get_color(),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Gradient updates")
    ax.set_ylabel("Validation value MSE (fixed 8,192-anchor subset)")
    ax.set_title("Validation curves; test data is not used for checkpoint selection")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.evaluation / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
