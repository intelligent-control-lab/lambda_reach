"""Train fresh SPARK monitors and select on validation before the held-out test."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "validation"))

import benchmark_spark_value as benchmark
from safety_value.monitoring.spark_bundle import load_bundle, sha256
from safety_value.monitoring.qualification import qualify_metrics


def select_lambda(evaluation, candidates, minimum_episodes):
    """Choose the lowest-MSE λ candidate that passes every validation gate."""
    if evaluation.get("split") != "validation" or not evaluation.get("complete"):
        raise ValueError("Selection requires a complete validation evaluation")
    checks = {}
    for model in candidates:
        metrics = evaluation["models"][str(model)]
        checks[str(model)] = qualify_metrics(metrics, {"episodes_min": minimum_episodes})
    eligible = [name for name, result in checks.items() if result["passed"]]
    selected = min(eligible, key=lambda name: (
        checks[name]["metrics"]["sample_value_error"],
        checks[name]["metrics"]["paper_false_safe_rate"], name)) if eligible else None
    return dict(selected=selected, checks=checks,
                rule="Among λ candidates passing all validation gates, lowest sample-value MSE; false-safe rate then path break ties")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=1638)
    parser.add_argument("--validation-episodes", type=int, default=410)
    parser.add_argument("--test-episodes", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--updates", type=int, default=2000)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="Train one seed instead of the default three")
    seeds.add_argument("--seeds", type=int, nargs="+", help="Training seeds; default: 0 1 2")
    parser.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True,
                        help="Also train supervised models on the same data and budget")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="Value-training device; rollout collection stays on CPU")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    training_seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else [0, 1, 2])
    if min(training_seeds) < 0 or len(set(training_seeds)) != len(training_seeds):
        parser.error("Training seeds must be distinct and nonnegative")
    if min(args.train_episodes, args.validation_episodes, args.test_episodes,
           args.steps, args.workers, args.updates, args.threads, args.bootstrap) < 1:
        parser.error("Counts must be positive")
    for path in (args.policy, args.config):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipe = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    recipe.update(policy_sha256=sha256(args.policy), config_sha256=sha256(args.config))
    recipe["training_seeds"] = training_seeds
    recipe["selection_rule"] = "Validation gates then lowest sample-value MSE; freeze before test evaluation"
    benchmark.save_json(output / "recipe.json", recipe)
    data = output / "data"
    benchmark.collect(SimpleNamespace(**{**vars(args), "output": data, "family": "avoid"}))
    models, lambda_models = [], []
    for method in (["lambda_reachability", "supervised"] if args.baselines else ["lambda_reachability"]):
        for seed in training_seeds:
            model = output / f"{method}_seed{seed}"
            benchmark.train(SimpleNamespace(data=data, output=model, method=method,
                                            seed=seed, updates=args.updates, threads=args.threads,
                                            device=args.device))
            models.append(model)
            if method == "lambda_reachability":
                lambda_models.append(model)
    benchmark.evaluate(SimpleNamespace(data=data, models=models, split="validation",
        output=output / "evaluation_validation", bootstrap=args.bootstrap, threads=args.threads))
    validation_path = output / "evaluation_validation/results.json"
    selection = select_lambda(json.loads(validation_path.read_text()), lambda_models, args.validation_episodes)
    selection["validation_sha256"] = sha256(validation_path)
    selection["test_used_for_selection"] = False
    if selection["selected"] is None:
        benchmark.save_json(output / "selection.json", selection)
        print(json.dumps(dict(status="validation_quality_failed", test_evaluated=False)), flush=True)
        return 2
    model = Path(selection["selected"])
    selected_seed = json.loads((model / "training_config.json").read_text())["seed"]
    checkpoint = model / "models" / f"last_seed_{selected_seed}.pt"
    selection["checkpoint_sha256"] = sha256(checkpoint)
    benchmark.save_json(output / "selection.json", selection)
    # This file and selected checkpoint identity exist before any test prediction.
    benchmark.evaluate(SimpleNamespace(data=data, models=models, split="test",
        output=output / "evaluation_test", bootstrap=args.bootstrap, threads=args.threads))
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed during held-out evaluation")
    # A bundle records an evaluated pair; evaluation completion is not a claim
    # that arbitrary application-specific accuracy requirements have been met.
    files = dict(policy=data / "policy.onnx", config=data / "deploy.yaml",
        value_checkpoint=checkpoint,
        value_config=model / "training_config.json",
        test_results=output / "evaluation_test" / "results.json")
    bundle = {"schema": "lambda.spark.monitor_bundle.v1", "family": "avoid",
              "robot": "UnitreeG1WholeBodyDynamic1Config"}
    bundle.update({name: {"path": str(path.relative_to(output)), "sha256": sha256(path)}
                   for name, path in files.items()})
    bundle_path = output / "bundle.json"
    benchmark.save_json(bundle_path, bundle)
    load_bundle(bundle_path)
    print(json.dumps({"bundle": str(bundle_path), "complete": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
