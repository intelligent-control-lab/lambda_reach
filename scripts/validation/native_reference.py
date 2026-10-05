"""Build and drive the original C++ headers with deterministic fixture IO."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_reference(rl_repo, mujoco_repo, output):
    output = Path(output)
    include = output / "include"
    include.mkdir(parents=True, exist_ok=True)
    sources = {}

    def write(relative, text):
        dest = include / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)

    # Keep the original managers and all math. Only offline IO and test access
    # differ. Never instantiate SDK channels or LED/audio clients in this test.
    write(
        "unitree/dds_wrapper/common/unitree_joystick.hpp",
        """#pragma once
namespace unitree::common { struct UnitreeJoystick {
float ly() const {return 0;} float lx() const {return 0;} float rx() const {return 0;}
}; }
""",
    )
    write(
        "unitree/robot/g1/audio/g1_audio_client.hpp",
        """#pragma once
#include <stdexcept>
namespace unitree::robot::g1 { struct AudioClient {
void Init() {throw std::runtime_error("Hardware IO forbidden in validation");}
void SetTimeout(float) {} int LedControl(int,int,int) {return -1;}
}; }
""",
    )
    for name in ["algorithms.h", "mocap_client.h", "safety_monitor.h"]:
        relative = Path("isaaclab/algorithms") / name
        source = rl_repo / "deploy/include" / relative
        text = source.read_text()
        sources[str(source)] = sha(source)
        if name == "algorithms.h":
            marker = "session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);"
            assert text.count(marker) == 1
            text = text.replace(
                marker, marker + "\n        session_options.SetIntraOpNumThreads(1);"
            )
        else:
            text = text.replace("\nprivate:", "\npublic:")
        if name == "safety_monitor.h":
            marker = "return std::chrono::duration<double>(elapsed).count();"
            assert text.count(marker) == 1
            text = text.replace(marker, "return reference_sim_time;")
            text = "namespace isaaclab { inline double reference_sim_time=0; }\n" + text
        write(relative, text)
    bridge = mujoco_repo / "simulate/src/unitree_sdk2_bridge.h"
    expression = re.search(
        r"mj_data_->ctrl\[i\] = m\.tau\(\).*?;", bridge.read_text(), re.DOTALL
    )
    assert expression is not None
    sources[str(bridge)] = sha(bridge)
    write(
        "reference_bridge.h",
        """#pragma once
struct ReferenceMotor {
    float target, stiffness, damping;
    float tau() const {return 0;} float dq() const {return 0;}
    float q() const {return target;} float kp() const {return stiffness;}
    float kd() const {return damping;}
};
inline void reference_control(mjData* mj_data_,const float* target,const float* kp,const float* kd) {
    const int num_motor_=29;
    for(int i=0;i<num_motor_;++i) {
        ReferenceMotor m{target[i],kp[i],kd[i]};
"""
        + expression.group(0)
        + "\n    }\n}\n",
    )
    for relative in [
        "isaaclab/manager/observation_manager.h",
        "isaaclab/manager/action_manager.h",
        "isaaclab/manager/manager_term_cfg.h",
        "isaaclab/envs/manager_based_rl_env.h",
        "isaaclab/envs/mdp/observations/observations.h",
        "isaaclab/envs/mdp/actions/joint_actions.h",
        "unitree_articulation.h",
    ]:
        source = rl_repo / "deploy/include" / relative
        sources[str(source)] = sha(source)
    mj_dir = Path(mujoco.__file__).parent
    ort = rl_repo / "deploy/thirdparty/onnxruntime-linux-x64-1.22.0"
    binary = output / "native_reference"
    driver = output / "driver.cpp"
    driver.write_bytes(Path(__file__).with_suffix(".cpp").read_bytes())
    sources[str(driver)] = sha(driver)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-pthread",
        "-I" + str(include),
        "-I" + str(rl_repo / "deploy/include"),
        "-I/usr/include/eigen3",
        "-I" + str(ort / "include"),
        "-I" + str(mj_dir / "include"),
        str(driver),
        str(ort / "lib/libonnxruntime.so.1.22.0"),
        str(next(mj_dir.glob("libmujoco.so.*"))),
        "-lyaml-cpp",
        "-lspdlog",
        "-lfmt",
        "-Wl,-rpath," + str(ort / "lib"),
        "-Wl,-rpath," + str(mj_dir),
        "-o",
        str(binary),
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "build.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Native build failed; see {output / 'build.log'}")
    manifest = {
        "source_sha256": sources,
        "build_command": command,
        "instrumentation": [
            "keyboard command supplied directly",
            "MoCap latest sample supplied directly",
            "SDK joystick and disabled LED IO replaced by fixtures",
            "ONNX Runtime intra-op threads set to 1",
            "safety latch uses simulation time",
            "original bridge torque expression extracted verbatim",
        ],
        "excluded": [
            "DDS delivery latency/jitter",
            "keyboard UI",
            "physical MoCap transport",
            "elastic suspension band",
            "hardware IO",
        ],
        "mujoco_version": mujoco.__version__,
        "native_onnxruntime_version": "1.22.0",
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return binary, manifest


class NativeReference:
    def __init__(self, binary, config, actor, scene, dt, value=None):
        self.process = subprocess.Popen(
            [
                str(binary),
                str(config),
                str(actor),
                str(scene),
                str(value) if value else "-",
                str(dt),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.nq, self.nv, self.ns, self.nobs = self.read(4, np.int32)
        self.has_value = value is not None

    def read(self, count, dtype=np.float64):
        size = int(count) * np.dtype(dtype).itemsize
        data = self.process.stdout.read(size)
        if len(data) != size:
            raise RuntimeError(
                f"Native process ended unexpectedly: {self.process.poll()}"
            )
        return np.frombuffer(data, dtype=dtype).copy()

    def send(self, op, *arrays):
        self.process.stdin.write(op.encode())
        for array in arrays:
            self.process.stdin.write(np.asarray(array).tobytes())
        self.process.stdin.flush()

    def state(self):
        self.send("F")
        row = self.read(self.nq + self.nv + self.ns + 3)
        self.time, self.ball_contact, self.torso_contact = row[-3:]
        return (
            row[: self.nq],
            row[self.nq : self.nq + self.nv],
            row[self.nq + self.nv : -3],
        )

    def ball(self, position, velocity):
        self.send("B", np.asarray([*position, *velocity], np.float64))

    def reset(self, qpos, qvel, *, history=True):
        self.send(
            "R" if history else "S",
            np.asarray(qpos, np.float64),
            np.asarray(qvel, np.float64),
        )

    def observe(self, command, relative_ball, *, reset=False):
        self.send(
            "I" if reset else "O",
            np.asarray(command, np.float32),
            np.asarray(relative_ball, np.float32),
        )
        result = self.read(self.nobs + 58 + (2 if self.has_value else 0), np.float32)
        return {
            "obs": result[: self.nobs],
            "action": result[self.nobs : self.nobs + 29],
            "target": result[self.nobs + 29 : self.nobs + 58],
            "value": float(result[-2]) if self.has_value else None,
            "signal": float(result[-1]) if self.has_value else None,
        }

    def step(self, target, substeps, *, synchronized=False):
        self.send(
            "C" if synchronized else "P",
            np.asarray(target, np.float32),
            np.array([substeps], np.int32),
        )

    def close(self):
        if self.process.poll() is None:
            self.send("Q")
        self.process.communicate(timeout=10)
        if self.process.returncode:
            raise RuntimeError(f"Native process exited {self.process.returncode}")
