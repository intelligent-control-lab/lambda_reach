"""Explicit engineering gates for held-out SPARK monitor accuracy."""

import json
import math
from pathlib import Path

from .spark_bundle import load_bundle


DEFAULT_LIMITS = {
    "episodes_min": 512,
    "temporal_recall_min": 0.95,
    "paper_false_safe_rate_max": 0.10,
    "false_alarm_rate_max": 0.15,
    "sample_value_error_max": 0.20,
}


def qualify_metrics(metrics, limits=None):
    """Gate point estimates; retain episode-bootstrap intervals for interpretation.

    Missing classes, non-finite estimates and missing uncertainty estimates
    cannot pass. These thresholds describe a SPARK test distribution, not
    paper equivalence or a hardware safety certificate.
    """
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    if set(limits) != set(DEFAULT_LIMITS):
        raise ValueError("Unknown monitor qualification limit")
    if any(not math.isfinite(value) or value < 0 for value in limits.values()):
        raise ValueError("Qualification limits must be finite and nonnegative")
    if limits["episodes_min"] < 1:
        raise ValueError("At least one held-out episode is required")
    checks = {}
    for name, limit in limits.items():
        metric, direction = name.rsplit("_", 1)
        value = metrics.get(metric)
        valid = isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        if metric in {"temporal_recall", "paper_false_safe_rate", "false_alarm_rate"}:
            valid = valid and value <= 1
        checks[metric] = bool(valid and (value >= limit if direction == "min" else value <= limit))
    for name in ("safe_frames", "unsafe_frames", "unsafe_segments"):
        value = metrics.get(name)
        checks[name] = isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    intervals = metrics.get("episode_bootstrap_95ci", {})
    if not isinstance(intervals, dict):
        intervals = {}
    for metric in ("temporal_recall", "paper_false_safe_rate", "false_alarm_rate", "sample_value_error"):
        interval = intervals.get(metric, [])
        checks[metric + "_uncertainty"] = (
            isinstance(interval, (list, tuple)) and len(interval) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in interval)
            and interval[0] <= interval[1]
        )
    return dict(passed=all(checks.values()), checks=checks, limits=limits, metrics=metrics,
                interpretation="Engineering point-estimate gate on held-out SPARK episodes; not paper equivalence")


def qualify_bundle(bundle_path, limits=None):
    files = load_bundle(bundle_path)
    bundle = json.loads(Path(bundle_path).read_text())
    evaluation = json.loads(files["test_results"].read_text())
    matching = [metrics for metrics in evaluation["models"].values()
                if metrics.get("checkpoint_sha256") == bundle["value_checkpoint"]["sha256"]]
    if len(matching) != 1:
        raise ValueError("Expected exactly one held-out result for this checkpoint")
    result = qualify_metrics(matching[0], limits)
    result.update(policy_sha256=bundle["policy"]["sha256"],
                  value_sha256=bundle["value_checkpoint"]["sha256"],
                  dataset_sha256=evaluation["dataset_sha256"])
    return result
