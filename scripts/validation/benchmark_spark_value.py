"""Train and evaluate safety values on independent built-in SPARK G1 rollouts.

The 29-DoF benchmark uses the paper's metrics; it is not the 37-DoF paper task.
Compact NPZ files contain research inputs/labels, not a new simulator interface.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def metric_module():
    path = ROOT / "safety_value/scripts/3_result_analysis/evaluate_inference.py"
    spec = importlib.util.spec_from_file_location("paper_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init_collector(family, steps, policy_path, config_path):
    global ENV, POLICY, OBSERVER
    from spark_pipeline.locomotion import make_locomotion_environment
    from spark_policy.control.whole_body.unitree_g1.locomotion import (
        UnitreeLocomotionPolicy,
    )

    from safety_value.monitoring.spark_monitor import ReachabilityObserver, SignalConfig

    ENV, robot = make_locomotion_environment(
        config_path=config_path,
        seed=0,
        max_steps=steps,
        enable_ball=family == "avoid",
    )
    POLICY = UnitreeLocomotionPolicy(
        robot, model_path=policy_path, config_path=config_path
    )
    OBSERVER = ReachabilityObserver(config=SignalConfig(profile=f"simulation_{family}"))


def collect_one(job):
    output, seed, steps, family = job
    rng = np.random.default_rng(seed)
    ENV.task.seed = seed
    ENV.task.episode_index = -1
    yaw = rng.uniform(-np.pi, np.pi)
    ENV.task.root_pose = np.r_[
        rng.uniform(-0.5, 0.5, 2), 0.793, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
    ]
    command = np.array([rng.uniform(-0.5, 1.0), 0.0, 0.0], np.float32)
    ENV.task.command = command
    previous = ENV.reset_output()
    POLICY.reset()
    OBSERVER.reset()
    rows = {
        k: []
        for k in (
            "obs",
            "signal",
            "event",
            "time",
            "tilt",
            "height",
            "distance",
            "ball_force",
            "torso_force",
        )
    }
    for _ in range(steps):
        action, info = POLICY.act(previous.agent_feedback, previous.task_info)
        following = ENV.step_output(action, info)
        f = following.agent_feedback
        values = {
            "obs": info["policy_obs"],
            "signal": OBSERVER(previous, info, following)["safety_signal"],
            "event": following.task_info["event_onset"],
            "time": f["time"],
            "tilt": f["base_tilt"],
            "height": f["base_height"],
            "distance": np.linalg.norm(f["ball_position_w"] - f["root_pose_wxyz"][:3])
            if family == "avoid"
            else 0.0,
            "ball_force": f["ball_contact_force"],
            "torso_force": f["torso_contact_force"],
        }
        for key, value in values.items():
            rows[key].append(value)
        previous = following
        if following.done:
            break
    arrays = {
        key: np.asarray(value, dtype=np.bool_ if key == "event" else np.float32)
        for key, value in rows.items()
    }
    arrays["obs"] = arrays["obs"].reshape(len(arrays["signal"]), -1)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError(f"Non-finite rollout {seed}")
    np.savez_compressed(
        output,
        **arrays,
        seed=seed,
        command=command,
        terminated=bool(previous.task_info["terminated"]),
    )
    return {
        "file": Path(output).name,
        "seed": seed,
        "frames": len(arrays["signal"]),
        "unsafe_frames": int(np.sum(arrays["signal"] > 0)),
        "terminated": bool(previous.task_info["terminated"]),
        "sha256": digest(output),
    }


def collect(args):
    from dataclasses import asdict
    import shutil

    from spark_pipeline.locomotion import runtime_versions

    # Freeze the actor before creating workers. Every split uses these same
    # bytes even if a training/export process later replaces its own output.
    for path in (args.policy, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=False)
    policy_path = args.output.resolve() / "policy.onnx"
    config_path = args.output.resolve() / "deploy.yaml"
    shutil.copyfile(args.policy, policy_path)
    shutil.copyfile(args.config, config_path)
    (args.output / "collector_snapshot.py").write_text(Path(__file__).read_text())
    init_collector(args.family, args.steps, policy_path, config_path)
    manifest = {
        "schema": "lambda.spark.value_benchmark.v1",
        "family": args.family,
        "robot": "UnitreeG1WholeBodyDynamic1Config",
        "physics_dt": 0.005,
        "control_dt": 0.02,
        "max_steps": args.steps,
        "signal": asdict(OBSERVER.config),
        "alignment": "policy_obs[t] with post-action signal[t]; legacy recorder convention",
        "events": "SPARK post-step ball-spawn flags; split at flag index as in existing adapter",
        "command": "constant per episode: vx uniform[-0.5,1.0], vy=0, yaw_rate=0",
        "reset": "xy uniform[-0.5,0.5], yaw uniform[-pi,pi], nominal joints and zero velocity",
        "randomization": "no mass/friction randomization or additional push impulses",
        "policy_sha256": POLICY.model_sha256,
        "policy_file": policy_path.name,
        "config_file": config_path.name,
        "config_sha256": POLICY.config.source_sha256,
        "model_sha256": digest(ENV.agent.mujoco_xml_path),
        "versions": runtime_versions(),
        "collector_sha256": digest(__file__),
        "splits": {},
        "complete": False,
    }
    # Disjoint seed ranges are frozen before generating any labels or predictions.
    sizes = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
        "test": args.test_episodes,
    }
    jobs, destinations = [], {}
    for split, count in sizes.items():
        directory = args.output / split
        directory.mkdir()
        offset = {"train": 100000, "validation": 200000, "test": 300000}[split]
        if not 1 <= count < 100000:
            raise ValueError("Each split must have 1..99999 episodes")
        manifest["splits"][split] = []
        for i in range(count):
            path = directory / f"episode_{offset + i}.npz"
            jobs.append((str(path), offset + i, args.steps, args.family))
            destinations[offset + i] = split
    save_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_collector,
        initargs=(args.family, args.steps, policy_path, config_path),
    ) as pool:
        for i, row in enumerate(pool.map(collect_one, jobs), 1):
            manifest["splits"][destinations[row["seed"]]].append(row)
            if i % 16 == 0 or i == len(jobs):
                save_json(args.output / "manifest.json", manifest)
                print(
                    f"Collected {i}/{len(jobs)} episodes in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    manifest["complete"] = True
    manifest["seconds"] = time.monotonic() - started
    save_json(args.output / "manifest.json", manifest)


class ValueEpisodes:
    """Linear-memory future suffixes, split by whole episode before balancing."""

    def __init__(self, root, split, *, balance=False, seed=0):
        import torch

        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest["complete"]:
            raise ValueError("Collection is incomplete")
        self.segments, self.index = [], []
        metric = metric_module()
        for row in manifest["splits"][split]:
            path = root / split / row["file"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"Dataset hash mismatch: {path}")
            with np.load(path) as episode:
                obs, signal, event = episode["obs"], episode["signal"], episode["event"]
            for start, end, _ in metric.segment_trajectory(event, len(signal)):
                if end <= start:
                    continue
                x, ell = obs[start : end + 1], signal[start : end + 1]
                target = np.maximum.accumulate(ell[::-1])[::-1].copy()
                index = len(self.segments)
                self.segments.append(
                    (
                        torch.from_numpy(x),
                        torch.from_numpy(ell),
                        torch.from_numpy(target),
                    )
                )
                self.index.extend((index, i) for i in range(len(x) - 1))
        labels = np.array([self.segments[s][2][i].item() <= 0 for s, i in self.index])
        self.raw_counts = {"safe": int(labels.sum()), "unsafe": int((~labels).sum())}
        if min(self.raw_counts.values()) == 0:
            raise ValueError(f"{split} lacks one safety class: {self.raw_counts}")
        if balance:
            rng = np.random.default_rng(seed)
            n = min(self.raw_counts.values())
            indices = np.r_[
                rng.choice(np.flatnonzero(labels), n, replace=False),
                rng.choice(np.flatnonzero(~labels), n, replace=False),
            ]
            rng.shuffle(indices)
            self.index = [self.index[i] for i in indices]
        self.input_dim = self.segments[0][0].shape[1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from safety_value.safety_analysis_algos.spark_dataset import EpisodeSample

        s, i = self.index[index]
        x, ell, target = self.segments[s]
        # Paper preprocessing treats the end of every event segment as terminal.
        return EpisodeSample(
            (
                x[i],
                x[i + 1],
                ell[i : i + 1],
                target[i : i + 1],
                (target[i : i + 1] <= 0).float(),
                x[i + 1 :],
                ell[i + 1 :],
            ),
            True,
        )


def train(args):
    import torch

    from safety_value.safety_analysis_algos.dpe.trainer import (
        DiscountedPolicyEvaluationTrainer,
    )
    from safety_value.safety_analysis_algos.lambda_reachability.trainer import (
        LambdaReachabilityTrainer,
    )
    from safety_value.safety_analysis_algos.supervised.trainer import SupervisedTrainer

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "benchmark_snapshot.py").write_text(Path(__file__).read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    # Same balanced anchors for every method and training seed.
    training = ValueEpisodes(args.data, "train", balance=True, seed=42)
    validation = ValueEpisodes(args.data, "validation")
    kwargs = {
        "input_dim": training.input_dim,
        "train_dataset": training,
        "test_dataset": validation,
        "device": args.device,
        "hidden_dims": [256, 256],
        "lr": 0.001,
        "batch_size": 256,
        "eval_steps": 500,
        "eval_batches": 8192,
        "train_data_dir": str(args.output / "train"),
        "plot_dir": str(args.output / "plots"),
    }
    extra = {}
    if args.method == "lambda_reachability":
        extra = {
            "lambda_param": 0.99,
            "max_horizon": 200,
            "delta": 0.99,
            "v_term": -1e6,
            "weight_main": 1.0,
            "weight_hinge_lb": 0.0,
            "weight_hinge_mono": 0.0,
            "weight_bce": 0.0,
            "target_tau": 0.05,
            "target_update_period": 10,
        }
        trainer = LambdaReachabilityTrainer(**kwargs, **extra)
    elif args.method == "supervised":
        trainer = SupervisedTrainer(**kwargs)
    else:
        extra = {
            "initial_lambda": 0.9,
            "lambda_max": 0.99,
            "lambda_increase_ratio": 0.1,
            "lambda_update_steps": 500,
            "loss_converge_tol": 0.01,
            "use_target_network": True,
            "target_update_steps": 10,
            "target_tau": 0.05,
        }
        trainer = DiscountedPolicyEvaluationTrainer(**kwargs, **extra)
    config = {
        "algorithm": args.method,
        "seed": args.seed,
        "input_dim": training.input_dim,
        "hidden_dims": [256, 256],
        "lr": 0.001,
        "batch": 256,
        "total_steps": args.updates,
        "device": args.device,
        "paper_training_lineage": "simulation",
        "source_manifest_sha256": digest(args.data / "manifest.json"),
        "policy_sha256": manifest["policy_sha256"],
        "config_sha256": manifest["config_sha256"],
        "signal": manifest["signal"],
        "balanced_training_samples": len(training),
        "raw_training_classes": training.raw_counts,
        "validation_classes": validation.raw_counts,
        "hyperparameters": extra,
        "checkpoint_rule": "last fixed-budget checkpoint; test never used for selection",
    }
    source_files = [
        Path(__file__),
        ROOT / "safety_value/safety_analysis_algos/model.py",
        ROOT / "safety_value/safety_analysis_algos/dataset.py",
        ROOT / "safety_value/safety_analysis_algos/loss.py",
        ROOT / "safety_value/safety_analysis_algos" / args.method / "trainer.py",
    ]
    config["implementation_sha256"] = {
        str(path.relative_to(ROOT)): digest(path) for path in source_files
    }
    save_json(args.output / "training_config.json", config)
    started = time.monotonic()
    trainer.train(
        total_steps=args.updates,
        ckpt_dir=str(args.output / "models"),
        log_every=100,
        save_last_only=True,
        run_suffix=f"_seed_{args.seed}",
    )
    save_json(
        args.output / "complete.json",
        {
            "complete": True,
            "steps": trainer.global_step,
            "seconds": time.monotonic() - started,
        },
    )


def summarize(rows):
    a = np.asarray(rows, float)
    sums = a.sum(axis=0)
    (
        frames,
        unsafe,
        safe,
        missed,
        alarm,
        correct,
        sqerr,
        segments,
        uns_seg,
        recall,
        detected,
        segmse,
    ) = sums
    ratio = lambda x, y: float(x / y) if y else None
    return {
        "frames": int(frames),
        "unsafe_frames": int(unsafe),
        "safe_frames": int(safe),
        "segments": int(segments),
        "unsafe_segments": int(uns_seg),
        "temporal_recall": ratio(recall, uns_seg),
        "detection_rate": ratio(detected, uns_seg),
        "sample_value_error": ratio(segmse, segments),
        "frame_weighted_value_mse": ratio(sqerr, frames),
        "paper_false_safe_rate": ratio(missed, unsafe),
        "false_alarm_rate": ratio(alarm, safe),
        "invariant_accuracy": ratio(correct, frames),
    }


def score_episode(episode, prediction, name, metric):
    signal, event = episode["signal"], episode["event"]
    row = np.zeros(12)
    segment_rows = []
    for i, (start, end, onset) in enumerate(
        metric.segment_trajectory(event, len(signal))
    ):
        ell, value = signal[start : end + 1], prediction[start : end + 1]
        target = np.maximum.accumulate(ell[::-1])[::-1]
        unsafe, predicts_unsafe = target > 0, value > 0
        m = metric.compute_segment_metrics(
            name, i, start, end, onset, signal, prediction
        )
        row += [
            len(ell),
            unsafe.sum(),
            (~unsafe).sum(),
            (unsafe & ~predicts_unsafe).sum(),
            (~unsafe & predicts_unsafe).sum(),
            (unsafe == predicts_unsafe).sum(),
            np.square(value - target).sum(),
            1,
            m.temporal_recall is not None,
            m.temporal_recall or 0.0,
            m.t_lead is not None and m.t_lead > 0,
            m.sample_value_error,
        ]
        segment_rows.append(
            {
                "episode": name,
                "segment": i,
                "start": int(start),
                "end": int(end),
                "t_unsafe": None if m.t_unsafe is None else int(m.t_unsafe),
                "t_lead": None if m.t_lead is None else int(m.t_lead),
                "temporal_recall": m.temporal_recall,
                "value_mse": m.sample_value_error,
            }
        )
    return row, segment_rows


def evaluate(args):
    import torch

    from safety_value.monitoring.spark_monitor import ValuePredictor

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete dataset")
    metric = metric_module()
    predictors = {str(path): ValuePredictor(path) for path in args.models}
    for value in predictors.values():
        if value.config["policy_sha256"] != manifest["policy_sha256"]:
            raise ValueError("Policy mismatch")
        if value.config.get("source_manifest_sha256") != digest(
            args.data / "manifest.json"
        ):
            raise ValueError("Training/evaluation manifest mismatch")
    results = {
        "split": args.split,
        "complete": False,
        "models": {},
        "dataset_sha256": digest(args.data / "manifest.json"),
    }
    results["metric_source_sha256"] = digest(metric.__file__)
    results["evaluation_source_sha256"] = digest(__file__)
    (args.output / "benchmark_snapshot.py").write_text(Path(__file__).read_text())
    for name, predictor in list(predictors.items()) + [
        ("always_safe", None),
        ("always_unsafe", None),
    ]:
        rows, segments = [], []
        for item in manifest["splits"][args.split]:
            path = args.data / args.split / item["file"]
            if digest(path) != item["sha256"]:
                raise ValueError("Changed evaluation episode")
            with np.load(path) as episode:
                if predictor is None:
                    prediction = np.full(
                        len(episode["signal"]), -1.0 if name == "always_safe" else 1.0
                    )
                else:
                    prediction = np.concatenate(
                        [
                            predictor(x)
                            for x in np.array_split(
                                episode["obs"], max(1, len(episode["obs"]) // 512)
                            )
                        ]
                    )
                row, detail = score_episode(episode, prediction, item["file"], metric)
            rows.append(row)
            segments.extend(detail)
        result = summarize(rows)
        result["episodes"] = len(rows)
        result["checkpoint_sha256"] = predictor.sha256 if predictor else None
        rng = np.random.default_rng(42)
        rows = np.asarray(rows)
        draws = [
            summarize(rows[rng.integers(0, len(rows), len(rows))])
            for _ in range(args.bootstrap)
        ]
        result["episode_bootstrap_95ci"] = {
            key: np.quantile(
                [d[key] for d in draws if d[key] is not None], [0.025, 0.975]
            ).tolist()
            for key in (
                "temporal_recall",
                "sample_value_error",
                "paper_false_safe_rate",
                "false_alarm_rate",
                "invariant_accuracy",
            )
            if any(d[key] is not None for d in draws)
        }
        results["models"][name] = result
        tag = Path(name).name
        np.save(args.output / f"{tag}_episode_metrics.npy", rows)
        save_json(args.output / f"{tag}_segments.json", segments)
        save_json(args.output / "results.json", results)
        print(name, json.dumps(result), flush=True)
    results["complete"] = True
    save_json(args.output / "results.json", results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--family", choices=["avoid"], default="avoid")
    c.add_argument("--policy", type=Path, required=True, help="Explicit SPARK-trained ONNX actor")
    c.add_argument("--config", type=Path, required=True, help="Matching exported deploy.yaml")
    c.add_argument("--train-episodes", type=int, default=1638)
    c.add_argument("--validation-episodes", type=int, default=410)
    c.add_argument("--test-episodes", type=int, default=512)
    c.add_argument("--steps", type=int, default=1000)
    c.add_argument("--workers", type=int, default=6)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument(
        "--method", choices=["lambda_reachability", "supervised", "dpe"], required=True
    )
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--updates", type=int, default=2000)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--device", default="cpu", help="Value-training device, e.g. cpu or cuda:0")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--models", type=Path, nargs="+", required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--split", choices=["validation", "test"], default="test")
    e.add_argument("--bootstrap", type=int, default=1000)
    e.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    {"collect": collect, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
