"""λ value prediction and explicitly configured research signals for SPARK."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


class ValuePredictor:
    def __init__(self, path, checkpoint=None):
        path = Path(path)
        self.is_onnx = path.suffix == ".onnx"
        if self.is_onnx:
            import json
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            self.model = ort.InferenceSession(str(path), sess_options=options,
                                              providers=["CPUExecutionProvider"])
            self.input_name = self.model.get_inputs()[0].name
            self.input_dim = self.model.get_inputs()[0].shape[-1]
            self.config = json.loads(self.model.get_modelmeta().custom_metadata_map.get(
                "lambda.training_config", "{}"))
            self.checkpoint = path
        else:
            from .value_model import load_model
            self.model, self.input_dim, self.config, self.checkpoint = load_model(path, checkpoint)
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def __call__(self, observation):
        obs = np.asarray(observation, np.float32)
        if obs.ndim == 1:
            obs = obs[None]
        if obs.ndim != 2 or obs.shape[1] != self.input_dim:
            raise ValueError(f"Value model requires observations of width {self.input_dim}")
        if not np.isfinite(obs).all():
            raise ValueError("Value model observation must be finite")
        if self.is_onnx:
            values = np.concatenate([self.model.run(None, {self.input_name: row[None]})[0]
                                     for row in obs]).reshape(-1)
        else:
            import torch
            with torch.inference_mode():
                values = self.model(torch.from_numpy(obs)).numpy().reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Value model returned a non-finite prediction")
        return values


@dataclass(frozen=True)
class SignalConfig:
    profile: str = "simulation_avoid"
    phi_max: float = float(np.pi / 4)
    clearance_min: float = .25
    danger_distance: float = .5
    ball_contact_threshold: float = .1
    body_contact_threshold: float = 1.
    ball_ground_filter: float = .15
    latch_duration: float = .25


class ReachabilityObserver:
    """Predict from the applied policy input and label the resulting state.

    This observer never changes the action. The signal convention is named in
    the run manifest; simulation and hardware proxies must not be conflated.
    """

    def __init__(self, value=None, *, config=None):
        self.value = value
        self.config = config or SignalConfig()
        if self.config.profile not in {"simulation_avoid", "simulation_push", "hardware_push"}:
            raise ValueError("Unknown signal profile")
        if min(self.config.phi_max, self.config.clearance_min, self.config.danger_distance) <= 0:
            raise ValueError("Signal normalization parameters must be positive")
        self.reset()

    def reset(self):
        self.last_collision_time = -float("inf")

    def signal(self, state):
        cfg = self.config
        terms = [(float(state["base_tilt"]) - cfg.phi_max) / cfg.phi_max]
        if cfg.profile != "hardware_push":
            terms += [1. if state["torso_contact_force"] > cfg.body_contact_threshold else -1.,
                      float(np.clip((cfg.clearance_min - state.get("terrain_clearance", state["base_height"])) /
                                    cfg.clearance_min, -1., 3.))]
        if cfg.profile == "simulation_avoid":
            ball = np.asarray(state["ball_position_w"])
            contact = state["ball_contact_force"] > cfg.ball_contact_threshold and ball[2] > cfg.ball_ground_filter
            if contact:
                self.last_collision_time = float(state["time"])
            latched = float(state["time"]) - self.last_collision_time < cfg.latch_duration
            distance = np.linalg.norm(ball - np.asarray(state["root_pose_wxyz"])[:3])
            terms.append(1. if contact or latched else float(np.clip(
                (cfg.danger_distance - distance) / cfg.danger_distance, -1., -1e-6)))
        return np.float32(max(terms))

    def __call__(self, previous, action_info, following):
        result = {"safety_signal": self.signal(following.agent_feedback)}
        if self.value is not None:
            expected_policy = getattr(self.value, 'config', {}).get('policy_sha256')
            if expected_policy and expected_policy != action_info.get('policy_sha256'):
                raise ValueError('The value checkpoint was trained for a different policy')
            result["safety_value"] = np.float32(self.value(action_info["policy_obs"])[0])
        return result


class ReachabilityViewerObserver(ReachabilityObserver):
    """Display the measured research signal and learned value in a SPARK viewer."""

    def __init__(self, agent, value=None, *, config=None):
        self.agent = agent
        super().__init__(value, config=config)

    def reset(self):
        super().reset()
        self.agent.set_viewer_simulation_info({
            "Safety signal l(x)": "waiting", "Reachability V(x)": "waiting",
            "Predicted state": "waiting",
        })

    def __call__(self, previous, action_info, following):
        result = super().__call__(previous, action_info, following)
        value = result.get("safety_value")
        self.agent.set_viewer_simulation_info({
            "Safety signal l(x)": f"{result['safety_signal']:+.3f}",
            "Reachability V(x)": "not loaded" if value is None else f"{value:+.3f}",
            "Predicted state": "not loaded" if value is None else ("UNSAFE" if value > 0 else "safe"),
        })
        return result
