"""Finish a running SPARK locomotion experiment, with an explicit walking gate.

Outputs are experiments, not a paper-reproduction or robot-safety certificate.
No deployment defaults or tracked repository files are changed by this script.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check_export(run):
    result = json.loads((run / "result.json").read_text())
    for name, key in (("policy.onnx", "policy_sha256"), ("deploy.yaml", "config_sha256"),
                      ("checkpoint.pt", "checkpoint_sha256")):
        with (run / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != result[key]:
            raise ValueError(f"Export hash mismatch: {run / name}")


def check_isaac_robustness(report, export, *, expected_seed=120051, minimum_episodes=16):
    """Require the declared disturbed cohort for the exact exported actor."""
    checks = dict(
        plant=report.get("family") == "avoid" and report.get("actuator_mode") == "implicit",
        self_collisions=report.get("allow_self_collision") is True,
        physics=report.get("physics_dt") == .005 and report.get("control_dt") == .02,
        corrected_ball=report.get("ball_max_angular_velocity_deg_s", 0.) >= math.degrees(60.),
        perturbations=report.get("domain_randomization") is True
        and report.get("observation_noise") is True and report.get("push_velocity") == [-.5, .5],
        ball=report.get("ball_enabled") is True and report.get("ball_inputs_masked") is False,
        cohort=report.get("seed") == expected_seed and report.get("steps", 0) >= 1000,
        config=report.get("config_sha256") == export["config_sha256"],
    )
    policies = list(report.get("policies", {}).values())
    checks["actor"] = len(policies) == 1 and policies[0].get("policy_sha256") == export["policy_sha256"]
    rows = policies[0].get("commands", []) if len(policies) == 1 else []
    checks["commands"] = len(rows) == 4 and {r.get("command_mps") for r in rows} == {0., .3, -.3, .6}
    checks["deterministic_actor"] = len(policies) == 1 and policies[0].get("action_noise_std") == 0.
    checks["episodes"] = bool(rows) and all(isinstance(r.get("episodes"), int)
        and r["episodes"] >= max(16, minimum_episodes) for r in rows)
    checks["fall_counts"] = bool(rows) and all(isinstance(r.get("falls"), int)
        and 0 <= r["falls"] <= r.get("episodes", -1) for r in rows)
    episodes = sum(r.get("episodes", 0) for r in rows) if checks["episodes"] else 0
    falls = sum(r["falls"] for r in rows) if checks["fall_counts"] else None
    fraction = falls / episodes if episodes and falls is not None else None
    checks["fall_fraction"] = fraction is not None and fraction <= .05
    return dict(passed=all(checks.values()), checks=checks, episodes=episodes,
                falls=falls, fall_fraction=fraction, max_fall_fraction=.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-root", type=Path, required=True)
    parser.add_argument("--spark-python", type=Path, required=True)
    parser.add_argument("--nominal-run", type=Path, required=True)
    parser.add_argument("--avoidance-run", type=Path,
                        help="Validate an existing checked avoidance export, then train its monitor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=14400)
    parser.add_argument("--avoidance-iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--nominal-extra-iterations", type=int, default=0,
                        help="Additional nominal training budget if qualification fails; zero disables retries")
    parser.add_argument("--nominal-chunk-iterations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--qualification-seed", type=int, default=120011,
                        help="Reserved quiet-test seed; ball and Isaac tests use +20 and +40")
    parser.add_argument("--avoidance-episodes", type=int, default=32,
                        help="Paired MuJoCo episodes per command, at least 32")
    parser.add_argument("--isaac-episodes-per-command", type=int, default=16,
                        help="Disturbed Isaac episodes per command, at least 16")
    parser.add_argument("--check-only", action="store_true", help="Check an existing export without further training")
    args = parser.parse_args()
    if min(args.wait_seconds, args.avoidance_iterations, args.num_envs, args.workers,
           args.nominal_chunk_iterations) < 1 or args.nominal_extra_iterations < 0:
        parser.error("Counts and timeout must be positive")
    if args.qualification_seed < 0 or args.avoidance_episodes < 32 or args.isaac_episodes_per_command < 16:
        parser.error("Qualification seed must be nonnegative; MuJoCo/Isaac cohorts require at least 32/16 episodes per command")
    if args.avoidance_run and args.check_only:
        parser.error("--check-only checks nominal locomotion; omit --avoidance-run")
    args.spark_root = args.spark_root.resolve()
    # Keep the venv entry point: resolving its symlink would select the base env.
    args.spark_python = Path(os.path.abspath(args.spark_python))
    nominal = args.nominal_run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="waiting_for_nominal", finished=False, walking_check_passed=False,
                 nominal_run=str(nominal), started_utc=datetime.now(timezone.utc).isoformat(),
                 qualification_cohort=dict(quiet_seed=args.qualification_seed,
                     ball_seed=args.qualification_seed + 20,
                     isaac_seed=args.qualification_seed + 40,
                     ball_episodes_per_command=args.avoidance_episodes,
                     isaac_episodes_per_command=args.isaac_episodes_per_command),
                 nominal_gate=dict(steps=1000, episodes_per_command=8,
                     commands=[0.0, 0.3, -0.3], max_falls=0,
                     max_standing_rmse_mps=.05, max_walking_rmse_mps=.2,
                     backends=["mujoco", "isaac"],
                     reset="seeded randomized XY/yaw and joint velocities; ball-free",
                     interpretation="engineering progression gate, not a paper threshold"))

    def status(**updates):
        state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "status.json", state)
        print(json.dumps(updates), flush=True)

    def run(command, log, *, cwd=ROOT, accepted_codes=(0,)):
        with log.open("w") as stream:
            result = subprocess.run([str(x) for x in command], cwd=cwd,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode not in accepted_codes:
            raise subprocess.CalledProcessError(result.returncode, command)
        return result.returncode

    offline = ["bash", ROOT / "scripts/validation/without_legacy_repos.sh"]
    isaac_execution = ["bwrap", "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    for hidden in (ROOT.parent / "unitree_rl_lab-lambda", ROOT.parent / "unitree_mujoco-lambda",
                   ROOT / "source", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spark/g1-locomotion-v1"):
        if hidden.exists():
            isaac_execution += ["--tmpfs", hidden]
    isaac_execution += ["--"]

    def isolated_training(command):
        prefix = ["bwrap", "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
        for hidden in (ROOT.parent / "unitree_rl_lab-lambda", ROOT.parent / "unitree_mujoco-lambda",
                       ROOT, Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spark/g1-locomotion-v1"):
            if hidden.exists():
                prefix += ["--tmpfs", hidden]
        return prefix + ["--chdir", args.spark_root, "--"] + command

    def check_walking(policy_run, name):
        destination = output / name
        destination.mkdir()
        reports = {}
        for backend in ("mujoco", "isaac"):
            prefix = offline
            if backend == "isaac":
                # The NVIDIA ground asset needs network access. Hide legacy
                # code/assets, while retaining SPARK's built-in USD cache.
                prefix = isaac_execution
            run(prefix + [args.spark_python,
                args.spark_root / "example/unitree_g1/locomotion/qualify.py",
                "--policy-run", policy_run, "--backends", backend,
                "--episodes", "8", "--steps", "1000", "--seed", str(args.qualification_seed),
                "--output", destination / backend],
                destination / f"{backend}.log", accepted_codes=(0, 2))
            reports[backend] = json.loads((destination / backend / "qualification.json").read_text())
        accepted = all(report["passed"] for report in reports.values())
        write_json(destination / "qualification.json", dict(passed=accepted, reports=reports,
                   gate=state["nominal_gate"]))
        return accepted

    try:
        status()
        deadline = time.monotonic() + args.wait_seconds
        while not (nominal / "result.json").exists():
            if (nominal / "failure.txt").exists():
                raise RuntimeError(f"Policy training failed; see {nominal / 'failure.txt'}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No completed export after {args.wait_seconds} seconds")
            time.sleep(30)
        check_export(nominal)
        status(status="checking_nominal")
        passed = check_walking(nominal, "nominal_check")
        remaining = args.nominal_extra_iterations
        attempt = 0
        while not passed and remaining and not args.check_only:
            attempt += 1
            iterations = min(remaining, args.nominal_chunk_iterations)
            extended = args.spark_root / "outputs/locomotion" / f"{output.name}_nominal_{attempt}"
            if extended.exists():
                raise FileExistsError(f"Refusing to overwrite nominal continuation: {extended}")
            status(status="extending_nominal", nominal_run=str(extended),
                   nominal_parent=str(nominal), additional_iterations=iterations,
                   nominal_remaining_budget=remaining)
            run(isolated_training([args.spark_python,
                args.spark_root / "example/unitree_g1/locomotion/train.py",
                "--resume", nominal / "checkpoint.pt", "--iterations", iterations,
                "--output", extended]), output / f"nominal_extension_{attempt}.log", cwd=args.spark_root)
            remaining -= iterations
            nominal = extended
            check_export(nominal)
            status(status="checking_nominal", nominal_run=str(nominal), nominal_remaining_budget=remaining)
            passed = check_walking(nominal, f"nominal_check_{attempt}")
        if not passed or args.check_only:
            status(status="nominal_check_passed" if passed else "nominal_quality_failed",
                   finished=True, walking_check_passed=passed,
                   next_step="Avoidance and monitor stages not run by this invocation")
            return 0 if passed else 2

        # Training output and its recipe live in SPARK, so the complete research
        # repository can remain hidden from the Isaac subprocess.
        if args.avoidance_run:
            avoidance = args.avoidance_run.resolve()
            status(status="loading_avoidance", avoidance_run=str(avoidance))
        else:
            avoidance = args.spark_root / "outputs/locomotion" / (output.name + "_avoidance")
            recipe = args.spark_root / "example/unitree_g1/locomotion/recipes/avoidance.json"
            if avoidance.exists():
                raise FileExistsError(f"Choose a fresh continuation name: {avoidance}")
            nominal_width = json.loads((nominal / "result.json").read_text())["input_dimension"]
            if nominal_width not in (480, 510, 512):
                raise ValueError(f"Unsupported nominal actor input dimension: {nominal_width}")
            transfer_args = ["--extend-ball-inputs"] if nominal_width == 480 else []
            command = isolated_training([args.spark_python,
                        args.spark_root / "example/unitree_g1/locomotion/train.py",
                        "--initialize-from", nominal / "checkpoint.pt", "--reset-ball-inputs",
                        "--ball-warmup-iterations", "0", "--num-envs", args.num_envs,
                        "--iterations", args.avoidance_iterations, "--recipe", recipe, "--output", avoidance,
                        *transfer_args])
            status(status="training_avoidance", avoidance_run=str(avoidance))
            run(command, output / "avoidance.log", cwd=args.spark_root)
        check_export(avoidance)
        status(status="checking_avoidance_locomotion")
        retained_walking = check_walking(avoidance, "avoidance_check")
        if not retained_walking:
            status(status="avoidance_quality_failed", finished=True, walking_check_passed=False,
                   next_step="Monitor training withheld because avoidance lost walking quality")
            return 2
        status(status="checking_physical_avoidance", walking_check_passed=retained_walking)
        run(offline + [args.spark_python,
            args.spark_root / "example/unitree_g1/locomotion/evaluate_avoidance.py",
            "--policy", avoidance / "policy.onnx", "--config", avoidance / "deploy.yaml",
            "--episodes", args.avoidance_episodes, "--steps", "1000", "--seed", args.qualification_seed + 20,
            "--workers", args.workers, "--output", output / "physical_avoidance"],
            output / "physical_avoidance.log", accepted_codes=(0, 2))
        ball_report = json.loads((output / "physical_avoidance/results.json").read_text())
        if not ball_report.get("complete") or not ball_report.get("passed"):
            status(status="physical_avoidance_quality_failed", finished=True,
                   avoidance_check_passed=False,
                   next_step="Monitor training withheld because physical avoidance did not pass its paired test")
            return 2
        status(status="checking_isaac_robustness")
        run(isaac_execution + [args.spark_python,
            args.spark_root / "example/unitree_g1/locomotion/diagnose_training.py",
            "--policy", avoidance / "policy.onnx", "--config", avoidance / "deploy.yaml",
            "--episodes-per-command", args.isaac_episodes_per_command, "--commands", "0", ".3", "-.3", ".6",
            "--steps", "1000", "--seed", args.qualification_seed + 40, "--ball", "--training-perturbations",
            "--output", output / "isaac_robustness.json"], output / "isaac_robustness.log")
        robustness = check_isaac_robustness(
            json.loads((output / "isaac_robustness.json").read_text()),
            json.loads((avoidance / "result.json").read_text()),
            expected_seed=args.qualification_seed + 40,
            minimum_episodes=args.isaac_episodes_per_command)
        write_json(output / "isaac_robustness_qualification.json", robustness)
        if not robustness["passed"]:
            status(status="isaac_robustness_quality_failed", finished=True,
                   next_step="Monitor training withheld because disturbed Isaac trials failed")
            return 2
        status(status="training_monitor", avoidance_check_passed=True)
        monitor_code = run(offline + [sys.executable, ROOT / "scripts/train_spark_monitor.py",
            "--policy", avoidance / "policy.onnx", "--config", avoidance / "deploy.yaml",
            "--output", output / "monitor", "--workers", args.workers, "--device", "cuda:0",
            "--train-episodes", "1638", "--validation-episodes", "410", "--test-episodes", "512"],
            output / "monitor.log", accepted_codes=(0, 2))
        if monitor_code == 2:
            status(status="monitor_validation_quality_failed", finished=True,
                   next_step="No value candidate passed validation; final test and online deployment withheld")
            return 2
        status(status="checking_monitor_accuracy")
        run(offline + [sys.executable, ROOT / "scripts/validation/qualify_spark_monitor.py",
            "--bundle", output / "monitor/bundle.json", "--output", output / "monitor/qualification.json"],
            output / "monitor_qualification.log", accepted_codes=(0, 2))
        value_qualification = json.loads((output / "monitor/qualification.json").read_text())
        if not value_qualification["passed"]:
            status(status="monitor_quality_failed", finished=True,
                   next_step="Inspect validation and held-out accuracy; a completed training job is not qualification")
            return 2
        status(status="checking_online_monitor")
        import h5py
        import numpy as np

        prediction_counts = {}
        for backend in ("mujoco", "isaac"):
            prefix = isaac_execution if backend == "isaac" else offline
            python = args.spark_python if backend == "isaac" else sys.executable
            trace = output / f"monitor/online_{backend}.hdf5"
            run(prefix + [python, ROOT / "safety_value/scripts/2_safety_analysis/run_monitor.py",
                "--bundle", output / "monitor/bundle.json", "--backend", backend, "--no-download",
                "--episodes", "4", "--steps", "1000", "--seed", "80000",
                "--velocity-command", ".3", "0", "0", "--output", trace],
                output / f"online_{backend}.log")
            prediction_counts[backend] = 0
            with h5py.File(trace) as trajectories:
                for episode in trajectories["episodes"].values():
                    values = episode["derived/safety_value"][:]
                    if not len(values) or not np.isfinite(values).all():
                        raise ValueError(f"Online {backend} monitor returned missing or non-finite predictions")
                    prediction_counts[backend] += len(values)
            if not prediction_counts[backend]:
                raise ValueError(f"Online {backend} monitor produced no predictions")
        manifest = json.loads((output / "monitor/data/manifest.json").read_text())
        falls = {split: sum(row["terminated"] for row in rows) / len(rows)
                 for split, rows in manifest["splits"].items()}
        status(status="complete" if retained_walking else "complete_with_policy_limitation",
               finished=True, bundle=str(output / "monitor/bundle.json"),
               walking_check_passed=retained_walking,
               online_prediction_counts=prediction_counts, rollout_fall_fractions=falls,
               monitor_accuracy_passed=True,
               isaac_robustness_passed=True,
               qualification_scope="Nominal commands in both simulators, paired physical ball avoidance in MuJoCo, disturbed Isaac trials, held-out monitor accuracy, and online execution in both simulators")
        return 0
    except Exception as error:
        status(status="failed", finished=True, error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
