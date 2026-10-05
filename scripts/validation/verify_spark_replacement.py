"""Compare SPARK against original Unitree C++ code under controlled simulation IO.

Outputs are experiments, not a certificate for DDS/hardware or all G1 tasks.
Run from lambda_reach with its .venv; the two original repos remain read-only.
"""

import argparse
import copy
import itertools
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from native_reference import NativeReference, build_reference, sha
from spark_pipeline.locomotion import make_locomotion_environment, runtime_versions
from spark_policy.control.whole_body.unitree_g1.locomotion import (
    RESOURCE_DIR,
    UnitreeLocomotionPolicy,
)

from safety_value.monitoring.spark_monitor import (
    ReachabilityObserver,
    SignalConfig,
    ValuePredictor,
)


def error(a, b):
    delta = np.asarray(a, float) - np.asarray(b, float)
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite comparison")
    return {
        "max_abs": float(np.abs(delta).max()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


def native_feedback(native, model):
    q, v, sensors = native.state()

    def sensor(name):
        item = model.sensor(name)
        return sensors[item.adr[0] : item.adr[0] + item.dim[0]].copy()

    pose = np.r_[sensor("frame_pos"), sensor("imu_quat").astype(np.float32)]
    # DDS motor/IMU payloads are floats; world ball/root positions are fixture inputs.
    omega = sensor("imu_gyro").astype(np.float32)
    f = {
        "time": native.time,
        "root_pose_wxyz": pose,
        "root_angular_velocity_b": omega,
        "root_linear_velocity_w": sensor("frame_vel"),
        "body_joint_pos": sensors[:29].astype(np.float32),
        "body_joint_vel": sensors[29:58].astype(np.float32),
        "base_height": float(pose[2]),
        "base_tilt": float(
            np.arccos(np.clip(1 - 2 * (pose[4] ** 2 + pose[5] ** 2), -1, 1))
        ),
        "torso_contact_force": native.torso_contact,
        "ball_contact_force": native.ball_contact,
    }
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "moving_ball_joint")
    if joint >= 0:
        ball_q, ball_v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        f.update(
            ball_position_w=q[ball_q : ball_q + 3],
            ball_velocity_w=v[ball_v : ball_v + 3],
        )
    return f, q, v


def relative_ball(f):
    return np.r_[
        f["ball_position_w"] - f["root_pose_wxyz"][:3],
        f["ball_velocity_w"] - f["root_linear_velocity_w"],
    ]


def add_reference_ball(scene, output):
    """Add the common physical stimulus independently of the SPARK agent loader."""
    scene_root = ET.parse(scene).getroot()
    robot_file = scene.parent / scene_root.find("include").attrib["file"]
    root = ET.parse(robot_file).getroot()
    root.find("compiler").set("meshdir", str((robot_file.parent / "meshes").resolve()))
    world = root.find("worldbody")
    for child in scene_root:
        if child.tag == "include":
            continue
        if child.tag == "worldbody":
            world.extend(copy.deepcopy(list(child)))
        else:
            root.append(copy.deepcopy(child))
    ball = ET.SubElement(world, "body", name="moving_ball", pos="0 0 -10")
    ET.SubElement(ball, "freejoint", name="moving_ball_joint")
    ET.SubElement(
        ball,
        "geom",
        name="moving_ball_geom",
        type="sphere",
        size=".1",
        mass=".05",
        rgba=".9 .3 .1 1",
        friction=".5 .005 .0001",
    )
    ET.ElementTree(root).write(output, encoding="unicode")
    return output


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.scene = args.mujoco_repo / "unitree_robots/g1/scene_29dof.xml"
        self.native_scene = {
            "push": self.scene,
            "avoid": add_reference_ball(self.scene, self.output / "reference_ball.xml"),
        }
        self.native_models = {
            k: mujoco.MjModel.from_xml_path(str(v))
            for k, v in self.native_scene.items()
        }
        self.binary, self.build = build_reference(
            args.rl_repo, args.mujoco_repo, self.output / "build"
        )
        self.value_path = self.output / "value.onnx"
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[2] / "deploy/tools/export_onnx.py"
                ),
                "--safety_value_dir",
                str(args.value),
                "--output",
                str(self.value_path),
            ],
            check=True,
        )
        self.value = ValuePredictor(args.value)
        self.report = {
            "versions": runtime_versions(),
            "reference": self.build,
            "value_sha256": self.value.sha256,
            "episodes_per_family": args.episodes,
            "steps_per_episode": args.steps,
            "control_dt": 0.02,
            "thresholds": {
                "shared_input_value_max_abs": 1e-5,
                "observation_max_abs": 1e-5,
                "action_max_abs": 1e-4,
                "same_model_open_loop_qpos_max_abs": 1e-5,
            },
            "fixed_state": {},
            "model_differences": {},
            "open_loop": {},
            "closed_loop": [],
        }

    def paths(self, family):
        base = (
            self.args.rl_repo
            / f"deploy/robots/g1_29dof/config/policy/velocity/{family}"
        )
        return base / "params/deploy.yaml", base / "exported/policy.onnx"

    def native(self, family, dt):
        config, actor = self.paths(family)
        return NativeReference(
            self.binary,
            config,
            actor,
            self.native_scene[family],
            dt,
            self.value_path if family == "avoid" else None,
        )

    def spark(self, family, profile="released", dt=0.002):
        env, robot = make_locomotion_environment(
            config_path=RESOURCE_DIR / f"configs/{family}.yaml",
            model_path=self.scene if profile == "released" else None,
            enable_ball=family == "avoid",
            max_steps=self.args.steps,
            agent_options={"dt": dt, "control_decimation": round(0.02 / dt)},
        )
        policy = UnitreeLocomotionPolicy(
            robot,
            config_path=RESOURCE_DIR / f"configs/{family}.yaml",
            model_path=self.paths(family)[1],
        )
        return env, policy

    def initial(self, model, config, seed=0):
        rng = np.random.default_rng(seed)
        q = model.qpos0.copy()
        q[:3] = [0, 0, 0.793]
        quat = Rotation.from_euler(
            "xyz",
            [
                rng.uniform(-0.025, 0.025),
                rng.uniform(-0.025, 0.025),
                rng.uniform(-0.2, 0.2),
            ],
        ).as_quat()
        q[3:7] = quat[[3, 0, 1, 2]]
        for name, position in zip(config.joint_names, config.default_position):
            q[model.jnt_qposadr[model.joint(name).id]] = position
        return q, np.zeros(model.nv)

    def fixed_states(self, family):
        native = self.native(family, 0.002)
        env, policy = self.spark(family)
        model = self.native_models[family]
        rng = np.random.default_rng(83)
        records = {
            key: []
            for key in [
                "native_obs",
                "spark_obs",
                "native_action",
                "spark_action",
                "shared_action",
                "native_target",
                "spark_target",
                "native_value",
                "shared_value",
                "spark_value",
                "ball_distance",
            ]
        }
        for i in range(256):
            q, v = self.initial(model, policy.config, i)
            quat = Rotation.from_euler(
                "xyz", rng.uniform([-0.4, -0.4, -3], [0.4, 0.4, 3])
            ).as_quat()
            q[3:7] = quat[[3, 0, 1, 2]]
            q[7:36] += rng.uniform(-0.05, 0.05, 29)
            v[:35] = rng.uniform(-0.3, 0.3, 35)
            native.reset(q, v, history=i == 0)
            f, _, _ = native_feedback(native, model)
            distance = [0.2, 0.5, 2.0, 4.9, 5.0, 5.1, 8.0, 15.0][(i // 32) % 8]
            direction = rng.normal(size=3)
            relative = direction / np.linalg.norm(direction) * distance
            f.update(
                ball_position_w=f["root_pose_wxyz"][:3] + relative,
                ball_velocity_w=rng.uniform(-6, 6, 3),
            )
            command = rng.uniform([-0.3, -0.2, -0.1], [0.3, 0.2, 0.1])
            n = native.observe(command, relative_ball(f), reset=i == 0)
            obs = policy.history.build(f, {"velocity_command": command})[0]
            own = policy.predict(obs)[0]
            shared = policy.predict(n["obs"])[0]
            target = policy.config.targets(own)
            # Replay the ORIGINAL previous action so observation mismatches do not
            # compound into the last-action term during this fixed-state test.
            policy.history.record_action(n["action"])
            values = {
                "native_obs": n["obs"],
                "spark_obs": obs,
                "native_action": n["action"],
                "spark_action": own,
                "shared_action": shared,
                "native_target": n["target"],
                "spark_target": target,
                "ball_distance": distance,
            }
            if family == "avoid":
                values.update(
                    native_value=n["value"],
                    shared_value=self.value(n["obs"])[0],
                    spark_value=self.value(obs)[0],
                )
            for key, value in values.items():
                records[key].append(value)
        native.close()
        env.agent.close_viewer()
        a = {k: np.asarray(v) for k, v in records.items() if v}
        np.savez_compressed(self.output / f"fixed_{family}.npz", **a)
        result = {
            "samples": len(a["native_obs"]),
            "observation": error(a["native_obs"], a["spark_obs"]),
            "actor_shared_input": error(a["native_action"], a["shared_action"]),
            "actor_own_input": error(a["native_action"], a["spark_action"]),
            "target_own_input": error(a["native_target"], a["spark_target"]),
        }
        if family == "avoid":
            near = a["ball_distance"] < 4.99
            result.update(
                base_480_observation=error(
                    a["native_obs"][:, :480], a["spark_obs"][:, :480]
                ),
                near_ball_observation=error(
                    a["native_obs"][near], a["spark_obs"][near]
                ),
                value_shared_input=error(a["native_value"], a["shared_value"]),
                value_own_input=error(a["native_value"], a["spark_value"]),
                value_sign_disagreements=int(
                    np.sum((a["native_value"] > 0) != (a["spark_value"] > 0))
                ),
            )
        self.report["fixed_state"][family] = result
        self.save()
        print("fixed_state", family, json.dumps(result), flush=True)

    def open_loop(self, dt, synchronized=False):
        native = self.native("push", dt)
        env, policy = self.spark("push", dt=dt)
        model = self.native_models["push"]
        q, v = self.initial(model, policy.config)
        native.reset(q, v)
        env.agent.reset({"root_pose_wxyz": q[:7], "joint_positions": q[7:]})
        qs, qn = [], []
        for i in range(200):
            target = policy.config.default_position + np.float32(
                0.015 * np.sin(i * 0.13)
            )
            native.step(target, round(0.02 / dt), synchronized=synchronized)
            env.agent.step(np.zeros(29), action_info={"target_actuated_pos": target})
            nq, nv, _ = native.state()
            qn.append(np.r_[nq, nv])
            qs.append(np.r_[env.agent.data.qpos, env.agent.data.qvel])
        name = str(dt) + ("_synchronized" if synchronized else "")
        np.savez_compressed(self.output / f"open_loop_{name}.npz", native=qn, spark=qs)
        result = {
            "qpos": error(np.asarray(qn)[:, : model.nq], np.asarray(qs)[:, : model.nq]),
            "qvel": error(np.asarray(qn)[:, model.nq :], np.asarray(qs)[:, model.nq :]),
            "first_step": error(qn[0], qs[0]),
            "steps": 200,
        }
        result["synchronized_reference_timing"] = synchronized
        self.report["open_loop"][name] = result
        native.close()
        env.agent.close_viewer()
        self.save()
        print("open_loop", dt, json.dumps(result), flush=True)

    def model_differences(self):
        original = self.native_models["push"]
        for profile in ["released", "spark"]:
            env, _ = self.spark("push", profile)
            other = env.agent.model
            result = {}
            for key in [
                "body_mass",
                "body_inertia",
                "dof_damping",
                "dof_armature",
                "dof_frictionloss",
                "geom_friction",
                "jnt_range",
                "actuator_ctrlrange",
                "actuator_gear",
            ]:
                a, b = getattr(original, key), getattr(other, key)
                result[key] = (
                    error(a, b)
                    if a.shape == b.shape
                    else {"shapes": [list(a.shape), list(b.shape)]}
                )
            result["total_mass"] = [
                float(original.body_mass.sum()),
                float(other.body_mass.sum()),
            ]
            self.report["model_differences"][profile] = result
            env.agent.close_viewer()
        self.save()

    def schedule(self, seed, family):
        rng = np.random.default_rng(seed + 12345)
        command = np.zeros((self.args.steps, 3), np.float32)
        command[:, 0] = [0.0, 0.2, -0.2, 0.3, 0.0][seed % 5]
        if family == "push" and seed % 5 == 4:
            command[:, 2] = 0.15
        launches = {}
        if family == "avoid":
            for step in range(0, self.args.steps, 175):
                angle = rng.uniform(-np.pi, np.pi)
                distance = rng.uniform(2.0, 5.0)
                position = np.r_[
                    distance * np.array([np.cos(angle), np.sin(angle)]),
                    rng.uniform(0.2, 1.2),
                ]
                speed = rng.uniform(3.0, 6.0)
                flight = distance / speed
                velocity = np.r_[
                    -speed * np.array([np.cos(angle), np.sin(angle)]),
                    (0.8 - position[2] + 0.5 * 9.81 * flight**2) / flight,
                ]
                launches[step] = (position, velocity)
        pushes = {250: np.array([0.0, 0.5 * (-1) ** seed, 0.0])}
        if seed % 5 == 0:
            pushes[500] = np.array([0.0, 4.0 * (-1) ** seed, 0.0])
        return command, launches, pushes

    def rollout(self, family, seed, profile, dt, trace_suffix=""):
        commands, launches, pushes = self.schedule(seed, family)
        observer = ReachabilityObserver(
            config=SignalConfig(profile=f"simulation_{family}")
        )
        records = {
            k: []
            for k in [
                "root",
                "joint",
                "obs",
                "action",
                "value",
                "signal",
                "native_signal",
            ]
        }
        native = None
        env = None
        if profile == "native":
            native = self.native(family, dt)
            model = self.native_models[family]
            # Configuration is read independently by the native executable.
            from spark_policy.control.whole_body.unitree_g1.locomotion import (
                LocomotionConfig,
            )

            config = LocomotionConfig.load(self.paths(family)[0])
            q, v = self.initial(model, config, seed)
            native.reset(q, v)
        else:
            env, policy = self.spark(family, profile, dt)
            model = env.agent.model
            q, v = self.initial(model, policy.config, seed)
            env.agent.reset(
                {"root_pose_wxyz": q[:7], "joint_positions": q[env.agent.q_indices]}
            )
            policy.reset()
        fell = False
        for step in range(self.args.steps):
            if step in launches:
                position, velocity = launches[step]
                if native:
                    native.ball(position, velocity)
                else:
                    env.agent.set_ball_state(position, velocity)
            if step in pushes:
                if native:
                    q, v, _ = native.state()
                    v[:3] += pushes[step]
                    native.reset(q, v, history=False)
                else:
                    env.agent.data.qvel[env.agent.root_v : env.agent.root_v + 3] += (
                        pushes[step]
                    )
                    mujoco.mj_forward(model, env.agent.data)
            if native:
                f, _, _ = native_feedback(native, model)
                rel = relative_ball(f) if family == "avoid" else np.zeros(6)
                action = native.observe(commands[step], rel, reset=step == 0)
                native.step(action["target"], round(0.02 / dt))
                after, q, v = native_feedback(native, model)
                value = action["value"]
            else:
                f = env.agent.get_feedback()
                _, info = policy.act(f, {"velocity_command": commands[step]})
                env.agent.step(np.zeros(29), action_info=info)
                after = env.agent.get_feedback()
                q, v = env.agent.data.qpos.copy(), env.agent.data.qvel.copy()
                action = {
                    "obs": info["policy_obs"],
                    "action": info["action_raw"],
                    "signal": np.nan,
                }
                value = (
                    float(self.value(action["obs"])[0]) if family == "avoid" else None
                )
            # Common post-action physical labels on the CURRENT generalized state;
            # the native deployment's different distance-proxy signal is saved too.
            after["root_pose_wxyz"] = q[:7].copy()
            after["base_height"] = float(q[2])
            after["base_tilt"] = float(
                np.arccos(np.clip(1 - 2 * (q[4] ** 2 + q[5] ** 2), -1, 1))
            )
            signal = float(observer.signal(after))
            records["root"].append(q[:7].copy())
            records["joint"].append(q[7:36].copy())
            records["obs"].append(action["obs"])
            records["action"].append(action["action"])
            records["signal"].append(signal)
            if family == "avoid":
                records["value"].append(value)
                records["native_signal"].append(action["signal"])
            fell = (
                after["base_height"] < 0.2
                or after["base_tilt"] > 0.8
                or (
                    family == "avoid"
                    and after["base_height"] < 0.4
                    and after["torso_contact_force"] > 1.0
                )
            )
            if fell:
                break
        if native:
            native.close()
        else:
            env.agent.close_viewer()
        data = {key: np.asarray(value) for key, value in records.items() if value}
        filename = f"{family}_{seed}_{profile}_{dt}{trace_suffix}.npz"
        np.savez_compressed(self.output / filename, **data)
        result = {
            "family": family,
            "seed": seed,
            "profile": profile,
            "physics_dt": dt,
            "steps": step + 1,
            "fell": bool(fell),
            "unsafe_frames": int((data["signal"] > 0).sum()),
            "minimum_height": float(data["root"][:, 2].min()),
            "trace": filename,
        }
        if family == "avoid":
            result["value_range"] = [
                float(data["value"].min()),
                float(data["value"].max()),
            ]
            result["value_nonfinite"] = int((~np.isfinite(data["value"])).sum())
            # Finite-horizon observed labels, cut at each obstacle launch.
            future_unsafe = np.zeros(len(data["signal"]), bool)
            cuts = sorted(
                {
                    0,
                    len(future_unsafe),
                    *[s for s in launches if s < len(future_unsafe)],
                }
            )
            for first, last in itertools.pairwise(cuts):
                future_unsafe[first:last] = (
                    np.maximum.accumulate(data["signal"][first:last][::-1])[::-1] > 0
                )
            predicted_unsafe = data["value"] > 0
            result["finite_horizon_confusion_unsafe_positive"] = {
                "tp": int((predicted_unsafe & future_unsafe).sum()),
                "fp": int((predicted_unsafe & ~future_unsafe).sum()),
                "tn": int((~predicted_unsafe & ~future_unsafe).sum()),
                "fn": int((~predicted_unsafe & future_unsafe).sum()),
            }
        return result, data

    def closed_loops(self):
        for family in ["push", "avoid"]:
            for seed in range(self.args.episodes):
                reference, baseline = self.rollout(family, seed, "native", 0.002)
                self.report["closed_loop"].append(reference)
                for profile, dt in [
                    ("released", 0.002),
                    ("released", 0.005),
                    ("spark", 0.005),
                ]:
                    result, data = self.rollout(family, seed, profile, dt)
                    count = min(len(data["root"]), len(baseline["root"]))
                    result["paired_steps"] = count
                    result["position_error"] = error(
                        data["root"][:count, :3], baseline["root"][:count, :3]
                    )
                    result["joint_error"] = error(
                        data["joint"][:count], baseline["joint"][:count]
                    )
                    result["fall_outcome_matches"] = result["fell"] == reference["fell"]
                    result["signal_sign_disagreement"] = float(
                        np.mean(
                            (data["signal"][:count] > 0)
                            != (baseline["signal"][:count] > 0)
                        )
                    )
                    if family == "avoid":
                        result["value_error"] = error(
                            data["value"][:count], baseline["value"][:count]
                        )
                        result["value_sign_disagreement"] = float(
                            np.mean(
                                (data["value"][:count] > 0)
                                != (baseline["value"][:count] > 0)
                            )
                        )
                    self.report["closed_loop"].append(result)
                self.save()
                rows = self.report["closed_loop"][-4:]
                print(
                    "closed_loop",
                    family,
                    seed,
                    [
                        (r["profile"], r["physics_dt"], r["steps"], r["fell"])
                        for r in rows
                    ],
                    flush=True,
                )

    def repeatability(self):
        self.report["reference_repeatability"] = {}
        seed = min(1, self.args.episodes - 1)
        for family in ["push", "avoid"]:
            _, repeated = self.rollout(family, seed, "native", 0.002, "_repeat")
            first = np.load(self.output / f"{family}_{seed}_native_0.002.npz")
            result = {k: error(first[k], repeated[k]) for k in repeated}
            self.report["reference_repeatability"][family] = result
        self.save()

    def finalize(self):
        rows = self.report["closed_loop"]
        summary = []
        for family in ["push", "avoid"]:
            for profile, dt in [
                ("native", 0.002),
                ("released", 0.002),
                ("released", 0.005),
                ("spark", 0.005),
            ]:
                selected = [
                    r
                    for r in rows
                    if r["family"] == family
                    and r["profile"] == profile
                    and r["physics_dt"] == dt
                ]
                row = {
                    "family": family,
                    "profile": profile,
                    "physics_dt": dt,
                    "episodes": len(selected),
                    "falls": sum(r["fell"] for r in selected),
                    "steps": sum(r["steps"] for r in selected),
                }
                if profile != "native":
                    count = sum(r["paired_steps"] for r in selected)
                    row["position_coordinate_rmse"] = float(
                        np.sqrt(
                            sum(
                                r["position_error"]["rmse"] ** 2 * r["paired_steps"]
                                for r in selected
                            )
                            / count
                        )
                    )
                    row["fall_outcome_agreement"] = sum(
                        r["fall_outcome_matches"] for r in selected
                    ) / len(selected)
                    if family == "avoid":
                        row["value_sign_disagreement"] = (
                            sum(
                                r["value_sign_disagreement"] * r["paired_steps"]
                                for r in selected
                            )
                            / count
                        )
                        row["value_rmse"] = float(
                            np.sqrt(
                                sum(
                                    r["value_error"]["rmse"] ** 2 * r["paired_steps"]
                                    for r in selected
                                )
                                / count
                            )
                        )
                if family == "avoid":
                    cm = {
                        k: sum(
                            r["finite_horizon_confusion_unsafe_positive"][k]
                            for r in selected
                        )
                        for k in ["tp", "fp", "tn", "fn"]
                    }
                    row["finite_horizon_confusion_unsafe_positive"] = cm
                    row["finite_horizon_unsafe_recall"] = cm["tp"] / max(
                        1, cm["tp"] + cm["fn"]
                    )
                summary.append(row)
        self.report["summary"] = summary
        fixed = self.report["fixed_state"]
        limits = self.report["thresholds"]
        gates = {
            "reference_repeatability": all(
                v["max_abs"] == 0
                for family in self.report["reference_repeatability"].values()
                for v in family.values()
            ),
            "actor_same_input": all(
                fixed[f]["actor_shared_input"]["max_abs"] <= limits["action_max_abs"]
                for f in fixed
            ),
            "value_same_input": fixed["avoid"]["value_shared_input"]["max_abs"]
            <= limits["shared_input_value_max_abs"],
            "push_observation_contract": fixed["push"]["observation"]["max_abs"]
            <= limits["observation_max_abs"],
            "avoid_observation_contract": fixed["avoid"]["observation"]["max_abs"]
            <= limits["observation_max_abs"],
            "native_bridge_control_timing": all(
                self.report["open_loop"][str(dt)]["qpos"]["max_abs"]
                <= limits["same_model_open_loop_qpos_max_abs"]
                for dt in [0.002, 0.005]
            ),
            "fall_outcomes_in_tested_scenarios": all(
                r.get("fall_outcome_matches", True) for r in rows
            ),
        }
        self.report["gates"] = gates
        self.report["strict_native_equivalence_pass"] = all(gates.values())
        self.report["complete"] = True
        self.report["working_source_sha256"] = {
            str(p): sha(p)
            for p in Path(__file__).parent.glob("*")
            if p.suffix in {".py", ".cpp"}
        }
        for name in [
            "spark_agent.simulation.mujoco.unitree_g1.unitree_locomotion_mujoco_agent",
            "spark_agent.simulation.mujoco.mujoco_agent",
            "spark_policy.control.whole_body.unitree_g1.locomotion.observations",
            "spark_policy.control.whole_body.unitree_g1.locomotion.policy",
            "spark_pipeline.locomotion",
            "safety_value.monitoring.spark_monitor",
        ]:
            source = Path(sys.modules[name].__file__)
            self.report["working_source_sha256"][str(source)] = sha(source)
        self.report["original_model_xml_sha256"] = {
            str(path): sha(path)
            for path in [self.scene, self.scene.parent / "g1_29dof.xml"]
        }
        self.report["original_xml_physics_dt"] = float(
            self.native_models["push"].opt.timestep
        )
        self.report["policy_sha256"] = {
            family: sha(self.paths(family)[1]) for family in ["push", "avoid"]
        }
        self.report["native_config_sha256"] = {
            family: sha(self.paths(family)[0]) for family in ["push", "avoid"]
        }
        self.report["limitations"] = [
            "This runs original C++ managers/observations/actors and the original bridge torque expression under deterministic fixture IO, not the original interactive executable.",
            "DDS latency/jitter, keyboard/FSM transitions, MoCap UDP freshness, LED/hardware IO and elastic suspension are excluded.",
            "Both physics paths use installed MuJoCo 3.8.0; equivalence with an older MuJoCo binary is not established.",
            "The original scene has no dynamic ball. Avoidance tests add the same physical ball to both paths and supply its state as ideal MoCap.",
            "Physical safety labels and fall criteria are shared test definitions. The original C++ proxy signal is also stored, but is not the physical training label.",
            "The value checkpoint was not retrained. Finite-horizon unsafe recall is descriptive for these scheduled scenarios, not a policy safety guarantee.",
        ]
        self.save()
        self.plot()
        lines = [
            "# SPARK replacement verification",
            "",
            "**Verdict: "
            + ("PASS" if all(gates.values()) else "NOT behaviorally equivalent")
            + "**",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
        lines += [
            f"| {name} | {'PASS' if passed else 'FAIL'} |"
            for name, passed in gates.items()
        ]
        lines += [
            "",
            "| Family | Profile | Physics dt | Falls / episodes | Root coordinate RMSE (m) | Value sign disagreement |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in summary:
            lines.append(
                f"| {r['family']} | {r['profile']} | {r['physics_dt']} | {r['falls']}/{r['episodes']} | "
                f"{r.get('position_coordinate_rmse', 0):.4f} | {r.get('value_sign_disagreement', 0):.2%} |"
            )
        lines += ["", "![Paired avoidance traces](comparison.png)", "", "## Scope", ""]
        lines += ["- " + item for item in self.report["limitations"]]
        (self.output / "REPORT.md").write_text("\n".join(lines) + "\n")
        print("Acceptance gates:", json.dumps(gates), flush=True)

    def plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for column, seed in enumerate([0, min(1, self.args.episodes - 1)]):
            for profile, dt, label in [
                ("native", 0.002, "Original C++ reference"),
                ("released", 0.002, "SPARK / original model / 2 ms"),
                ("spark", 0.005, "SPARK / built-in model / 5 ms"),
            ]:
                data = np.load(self.output / f"avoid_{seed}_{profile}_{dt}.npz")
                time = np.arange(1, len(data["root"]) + 1) * 0.02
                axes[0, column].plot(time, data["root"][:, 1], label=label, lw=1)
                axes[1, column].plot(time, data["value"], label=label, lw=1)
            axes[0, column].set_title(f"Avoidance seed {seed}")
            axes[0, column].set_ylabel("Root y (m)")
            axes[1, column].set_ylabel("Learned value V(x)")
            axes[1, column].set_xlabel("Simulation time (s)")
            axes[1, column].axhline(0, color="black", ls="--", lw=0.7)
            for ax in axes[:, column]:
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.savefig(self.output / "comparison.png", dpi=160)
        plt.close(fig)

    def save(self):
        (self.output / "results.json").write_text(
            json.dumps(self.report, indent=2) + "\n"
        )


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rl-repo", type=Path, default=root.parent / "unitree_rl_lab-lambda"
    )
    p.add_argument(
        "--mujoco-repo", type=Path, default=root.parent / "unitree_mujoco-lambda"
    )
    p.add_argument(
        "--value",
        type=Path,
        default=root
        / "logs/safety_analysis/spark_29dof/results/lambda_reachability_migrated",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()
    if args.episodes < 1 or args.steps < 1:
        p.error("episodes and steps must be positive")
    e = Experiment(args)
    e.model_differences()
    for family in ["push", "avoid"]:
        e.fixed_states(family)
    for dt in [0.002, 0.005]:
        e.open_loop(dt)
        e.open_loop(dt, synchronized=True)
    e.closed_loops()
    e.repeatability()
    e.finalize()
    print("Results:", e.output, flush=True)
    return 0 if e.report["strict_native_equivalence_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
