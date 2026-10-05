"""Regression checks for safety-metric direction and episode-held-out learning."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).resolve().parents[1] / "scripts/validation/benchmark_spark_value.py"
)
SPEC = importlib.util.spec_from_file_location("spark_value_benchmark", PATH)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_false_safe_is_paper_positive_not_alarm():
    episode = {"signal": np.array([-1.0, -1.0, 1.0, -1.0]), "event": np.zeros(4, bool)}
    metrics = benchmark.metric_module()
    safe, _ = benchmark.score_episode(episode, np.full(4, -1.0), "test", metrics)
    alarm, _ = benchmark.score_episode(episode, np.full(4, 1.0), "test", metrics)
    safe, alarm = benchmark.summarize([safe]), benchmark.summarize([alarm])
    assert safe["paper_false_safe_rate"] == 1
    assert safe["false_alarm_rate"] == 0
    assert safe["temporal_recall"] == 0
    assert alarm["paper_false_safe_rate"] == 0
    assert alarm["false_alarm_rate"] == 1
    assert alarm["temporal_recall"] == 1


def test_event_boundary_prevents_future_collision_leakage(tmp_path):
    episode = {
        "signal": np.array([-1.0, -1.0, -1.0, 1.0]),
        "event": np.array([0, 0, 1, 0]),
    }
    row, segments = benchmark.score_episode(
        episode, np.array([-1.0, -1.0, 1.0, 1.0]), "test", benchmark.metric_module()
    )
    result = benchmark.summarize([row])
    assert len(segments) == 2
    assert result["safe_frames"] == 2
    assert result["unsafe_frames"] == 2
    assert result["paper_false_safe_rate"] == 0
    assert result["invariant_accuracy"] == 1
    benchmark.save_json(tmp_path / "segments.json", segments)
    assert len(json.loads((tmp_path / "segments.json").read_text())) == 2


def test_whole_episode_split_and_training_balance(tmp_path):
    manifest = {"complete": True, "splits": {}}
    for split, marker in (("train", 1), ("test", 9)):
        (tmp_path / split).mkdir()
        path = tmp_path / split / "episode.npz"
        np.savez(
            path,
            obs=np.full((6, 2), marker, np.float32),
            signal=np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0], np.float32),
            event=np.zeros(6, bool),
        )
        manifest["splits"][split] = [
            {"file": path.name, "sha256": benchmark.digest(path)}
        ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    training = benchmark.ValueEpisodes(tmp_path, "train", balance=True)
    testing = benchmark.ValueEpisodes(tmp_path, "test")
    assert len(training) == 4
    assert sum(item[4].item() for item in training) == 2
    assert all((item[0].numpy() == 1).all() for item in training)
    assert all((item[0].numpy() == 9).all() for item in testing)
    assert all(item.future_terminal for item in training)
    # Mutated data cannot silently reuse the original split/provenance.
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark.ValueEpisodes(tmp_path, "test")
