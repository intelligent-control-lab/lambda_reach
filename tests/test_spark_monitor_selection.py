"""Keep monitor selection on validation and withhold unsafe candidates."""

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "monitor_training", Path(__file__).parents[1] / "scripts/train_spark_monitor.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


def metrics(error=.1, false_safe=.04):
    result = dict(episodes=410, temporal_recall=.98, paper_false_safe_rate=false_safe,
                  false_alarm_rate=.08, sample_value_error=error,
                  safe_frames=1000, unsafe_frames=100, unsafe_segments=10)
    result["episode_bootstrap_95ci"] = {name: [result[name] * .9, result[name] * 1.1]
        for name in ("temporal_recall", "paper_false_safe_rate", "false_alarm_rate", "sample_value_error")}
    return result


def test_selection_rejects_low_error_but_missed_hazards_and_test_data():
    evaluation = dict(complete=True, split="validation",
                      models={"unsafe": metrics(.01, .5), "qualified": metrics(.15)})
    result = training.select_lambda(evaluation, [Path("unsafe"), Path("qualified")], 410)
    assert result["selected"] == "qualified"
    evaluation["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        training.select_lambda(evaluation, [Path("qualified")], 410)


@pytest.mark.parametrize("passes_validation", [False, True])
def test_test_evaluation_requires_frozen_qualified_selection(tmp_path, monkeypatch, passes_validation):
    policy, config = tmp_path / "actor.onnx", tmp_path / "deploy.yaml"
    policy.write_bytes(b"actor")
    config.write_bytes(b"config")
    output = tmp_path / "experiment"
    evaluated = []

    def train(args):
        args.output.mkdir()
        (args.output / "models").mkdir()
        (args.output / "models" / f"last_seed_{args.seed}.pt").write_bytes(bytes([args.seed]))
        (args.output / "training_config.json").write_text(json.dumps(dict(seed=args.seed)))

    def evaluate(args):
        evaluated.append(args.split)
        if args.split == "test":
            selection = json.loads((output / "selection.json").read_text())
            assert selection["selected"].endswith("lambda_reachability_seed1")
            assert selection["checkpoint_sha256"]
            assert selection["test_used_for_selection"] is False
            # Test rankings deliberately reverse validation rankings.
        args.output.mkdir()
        scores = {str(model): metrics(.15 if ("seed0" in str(model)) == (args.split == "validation") else .1,
                                      .04 if passes_validation else .4) for model in args.models}
        (args.output / "results.json").write_text(json.dumps(dict(complete=True, split=args.split, models=scores)))

    monkeypatch.setattr(training.benchmark, "train", train)
    monkeypatch.setattr(training.benchmark, "evaluate", evaluate)
    monkeypatch.setattr(training, "load_bundle", lambda path: None)
    # Dataset copies are simulated here; production collection freezes them.
    def collect(args):
        args.output.mkdir()
        (args.output / "policy.onnx").write_bytes(policy.read_bytes())
        (args.output / "deploy.yaml").write_bytes(config.read_bytes())
    monkeypatch.setattr(training.benchmark, "collect", collect)
    monkeypatch.setattr(training.sys, "argv", ["train_spark_monitor.py", "--policy", str(policy),
        "--config", str(config), "--output", str(output), "--seeds", "0", "1", "--no-baselines"])
    assert training.main() == (0 if passes_validation else 2)
    assert evaluated == (["validation", "test"] if passes_validation else ["validation"])
    assert (output / "bundle.json").exists() == passes_validation
    if passes_validation:
        bundle = json.loads((output / "bundle.json").read_text())
        assert "lambda_reachability_seed1" in bundle["value_checkpoint"]["path"]
