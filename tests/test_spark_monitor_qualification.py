"""A completed value-training job alone must not pass the accuracy gate."""

import copy
import pytest

from safety_value.monitoring.qualification import qualify_metrics


def passing_metrics():
    values = dict(episodes=512, temporal_recall=.98, paper_false_safe_rate=.05,
                  false_alarm_rate=.06, sample_value_error=.12,
                  safe_frames=10000, unsafe_frames=2000, unsafe_segments=50)
    values["episode_bootstrap_95ci"] = {name: [values[name] - .01, values[name] + .01]
        for name in ("temporal_recall", "paper_false_safe_rate", "false_alarm_rate", "sample_value_error")}
    return values


def test_valid_heldout_metrics_pass_with_uncertainty_retained():
    metrics = passing_metrics()
    report = qualify_metrics(metrics)
    assert report["passed"] and report["metrics"] == metrics


@pytest.mark.parametrize("change", [
    dict(episodes=256), dict(temporal_recall=.94), dict(paper_false_safe_rate=.11),
    dict(false_alarm_rate=1.), dict(sample_value_error=.21), dict(unsafe_frames=0),
    dict(safe_frames=0), dict(unsafe_segments=0), dict(temporal_recall=float("nan")),
    dict(paper_false_safe_rate=None), dict(episode_bootstrap_95ci={}),
    dict(episode_bootstrap_95ci=None), dict(false_alarm_rate=-.1), dict(temporal_recall=1.1),
])
def test_inaccurate_or_uninformative_results_fail(change):
    metrics = passing_metrics()
    metrics.update(change)
    assert not qualify_metrics(metrics)["passed"]


def test_limits_are_explicit_and_validated():
    metrics = passing_metrics()
    before = copy.deepcopy(metrics)
    assert not qualify_metrics(metrics, {"sample_value_error_max": .10})["passed"]
    assert metrics == before
    with pytest.raises(ValueError):
        qualify_metrics(metrics, {"unknown": 1})
    with pytest.raises(ValueError):
        qualify_metrics(metrics, {"episodes_min": 0})
