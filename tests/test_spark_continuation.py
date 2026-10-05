"""Progression must stop before learning a monitor for a rejected actor."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failed_stage,extra,nominal_width", [
    ("nominal", 0, 510), ("avoidance", 0, 510), ("nominal", 4, 510),
    ("avoidance", 0, 480), ("avoidance", 0, 512)])
def test_failed_walking_gate_stops_continuation(tmp_path, monkeypatch, failed_stage, extra, nominal_width):
    path = Path(__file__).parents[1] / "scripts/finish_spark_retraining.py"
    spec = importlib.util.spec_from_file_location("continuation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nominal = tmp_path / "nominal"
    nominal.mkdir()
    (nominal / "result.json").write_text(json.dumps({"input_dimension": nominal_width}))
    output = tmp_path / "continuation"
    spark = tmp_path / "spark"
    (spark / "outputs/locomotion").mkdir(parents=True)
    monkeypatch.setattr(module, "check_export", lambda run: None)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        if any(str(item).endswith("qualify.py") for item in command):
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir(parents=True)
            is_nominal = "nominal_check" in str(destination)
            passed = is_nominal and failed_stage == "avoidance"
            (destination / "qualification.json").write_text(json.dumps({"passed": passed}))
            return SimpleNamespace(returncode=0 if passed else 2)
        assert any(str(item).endswith("locomotion/train.py") for item in command)
        if "--resume" in command:
            assert extra and "--initialize-from" not in command
        else:
            assert "--initialize-from" in command
            assert str(command[command.index("--recipe") + 1]).endswith("recipes/avoidance.json")
            assert ("--extend-ball-inputs" in command) == (nominal_width == 480)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    monkeypatch.setattr(module.sys, "argv", [str(path), "--spark-root", str(spark),
        "--spark-python", "/test/python", "--nominal-run", str(nominal), "--output", str(output),
        "--nominal-extra-iterations", str(extra), "--nominal-chunk-iterations", "3"])
    assert module.main() == 2
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == f"{failed_stage}_quality_failed"
    assert status["finished"] and not status["walking_check_passed"]
    assert len(commands) == (8 if extra else 2 if failed_stage == "nominal" else 5)
    if extra:
        updates = [int(command[command.index("--iterations") + 1]) for command in commands if "--resume" in command]
        assert updates == [3, 1]
        assert status["nominal_remaining_budget"] == 0
    assert not any("train_spark_monitor.py" in str(command) for command in commands)


def robustness_report():
    return dict(family="avoid", actuator_mode="implicit", physics_dt=.005, control_dt=.02,
        allow_self_collision=True,
        ball_max_angular_velocity_deg_s=5729.57795, domain_randomization=True,
        observation_noise=True, push_velocity=[-.5, .5], ball_enabled=True,
        ball_inputs_masked=False, seed=120051, steps=1000, config_sha256="config",
        policies={"actor": dict(policy_sha256="actor", action_noise_std=0.,
            commands=[dict(command_mps=c, episodes=16, falls=0) for c in (0., .3, -.3, .6)])})


@pytest.mark.parametrize("failed_stage", ["physical_avoidance", "isaac_robustness", "monitor_validation", "monitor", "complete", "complete_existing", "complete_larger"])
def test_failed_safety_gate_stops_continuation(tmp_path, monkeypatch, failed_stage):
    import h5py  # Load platform probes before replacing subprocess.run below.

    path = Path(__file__).parents[1] / "scripts/finish_spark_retraining.py"
    spec = importlib.util.spec_from_file_location("value_continuation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nominal = tmp_path / "nominal"
    nominal.mkdir()
    (nominal / "result.json").write_text(json.dumps({"input_dimension": 512}))
    spark = tmp_path / "spark"
    (spark / "outputs/locomotion").mkdir(parents=True)
    output = tmp_path / "continuation"
    existing = tmp_path / "existing_avoidance"
    if failed_stage == "complete_existing":
        existing.mkdir()
        (existing / "result.json").write_text(json.dumps(dict(policy_sha256="actor", config_sha256="config")))
    monkeypatch.setattr(module, "check_export", lambda run: None)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index("--output") + 1])
        if any(str(item).endswith("run_monitor.py") for item in command):
            with h5py.File(destination, "w") as trace:
                trace.create_dataset("episodes/000/derived/safety_value", data=[-.2, .1, -.1])
            return SimpleNamespace(returncode=0)
        if any(str(item).endswith("diagnose_training.py") for item in command):
            assert "--ball" in command and "--training-perturbations" in command
            report = robustness_report()
            if failed_stage == "complete_larger":
                assert command[command.index("--seed") + 1] == "130051"
                assert command[command.index("--episodes-per-command") + 1] == "32"
                report["seed"] = 130051
                for row in report["policies"]["actor"]["commands"]:
                    row["episodes"] = 32
            if failed_stage == "isaac_robustness":
                report["policies"]["actor"]["commands"][0]["falls"] = 4
            destination.write_text(json.dumps(report))
            return SimpleNamespace(returncode=0)
        if any(str(item).endswith("qualify_spark_monitor.py") for item in command):
            passed = failed_stage != "monitor"
            destination.write_text(json.dumps(dict(passed=passed)))
            return SimpleNamespace(returncode=0 if passed else 2)
        destination.mkdir(parents=True, exist_ok=True)
        if any(str(item).endswith("locomotion/train.py") for item in command):
            (destination / "result.json").write_text(json.dumps(dict(policy_sha256="actor", config_sha256="config")))
        if any(str(item).endswith("qualify.py") for item in command):
            if failed_stage == "complete_larger":
                assert command[command.index("--seed") + 1] == "130011"
            (destination / "qualification.json").write_text(json.dumps(dict(passed=True)))
        if any(str(item).endswith("evaluate_avoidance.py") for item in command):
            assert command[command.index("--episodes") + 1] == ("128" if failed_stage == "complete_larger" else "32")
            if failed_stage == "complete_larger":
                assert command[command.index("--seed") + 1] == "130031"
            passed = failed_stage != "physical_avoidance"
            (destination / "results.json").write_text(json.dumps(dict(complete=True, passed=passed)))
            return SimpleNamespace(returncode=0 if passed else 2)
        if any(str(item).endswith("train_spark_monitor.py") for item in command):
            assert command[command.index("--test-episodes") + 1] == "512"
            if failed_stage == "monitor_validation":
                return SimpleNamespace(returncode=2)
            (destination / "data").mkdir()
            (destination / "data/manifest.json").write_text(json.dumps(dict(
                splits={split: [dict(terminated=False)] for split in ("train", "validation", "test")})))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    monkeypatch.setattr(module.sys, "argv", [str(path), "--spark-root", str(spark),
        "--spark-python", "/test/python", "--nominal-run", str(nominal), "--output", str(output),
        *(["--qualification-seed", "130011", "--avoidance-episodes", "128", "--isaac-episodes-per-command", "32"] if failed_stage == "complete_larger" else []),
        *(["--avoidance-run", str(existing)] if failed_stage == "complete_existing" else [])])
    assert module.main() == (0 if failed_stage.startswith("complete") else 2)
    status = json.loads((output / "status.json").read_text())
    if failed_stage.startswith("complete"):
        assert status["status"] == "complete" and status["finished"]
        assert status["isaac_robustness_passed"] and status["monitor_accuracy_passed"]
        assert status["online_prediction_counts"] == dict(mujoco=3, isaac=3)
        online = [c for c in commands if any(str(v).endswith("run_monitor.py") for v in c)]
        assert {c[c.index("--backend") + 1] for c in online} == {"mujoco", "isaac"}
        if failed_stage == "complete_existing":
            assert status["avoidance_run"] == str(existing)
            assert not any(any(str(v).endswith("locomotion/train.py") for v in c) for c in commands)
        return
    assert status["status"] == f"{failed_stage}_quality_failed" and status["finished"]
    assert not any("run_monitor.py" in str(command) for command in commands)
    if failed_stage in {"physical_avoidance", "isaac_robustness"}:
        assert not any("train_spark_monitor.py" in str(command) for command in commands)


@pytest.mark.parametrize("invalid", [None, "wrong_actor", "no_push", "ball_masked", "old_ball_limit",
    "short_cohort", "wrong_seed", "missing_command", "few_episodes", "too_many_falls", "negative_falls", "noise", "self_collisions_disabled"])
def test_isaac_robustness_requires_exact_actor_and_declared_conditions(invalid):
    path = Path(__file__).parents[1] / "scripts/finish_spark_retraining.py"
    spec = importlib.util.spec_from_file_location("robustness_continuation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = robustness_report()
    actor = report["policies"]["actor"]
    if invalid == "wrong_actor": actor["policy_sha256"] = "different"
    if invalid == "no_push": report["push_velocity"] = [0., 0.]
    if invalid == "ball_masked": report["ball_inputs_masked"] = True
    if invalid == "old_ball_limit": report["ball_max_angular_velocity_deg_s"] = 100.
    if invalid == "short_cohort": report["steps"] = 999
    if invalid == "wrong_seed": report["seed"] += 1
    if invalid == "missing_command": actor["commands"].pop()
    if invalid == "few_episodes": actor["commands"][0]["episodes"] = 15
    if invalid == "too_many_falls": actor["commands"][0]["falls"] = 4
    if invalid == "negative_falls": actor["commands"][0]["falls"] = -1
    if invalid == "noise": actor["action_noise_std"] = .2
    if invalid == "self_collisions_disabled": report["allow_self_collision"] = False
    result = module.check_isaac_robustness(report, dict(policy_sha256="actor", config_sha256="config"))
    assert result["passed"] == (invalid is None)
