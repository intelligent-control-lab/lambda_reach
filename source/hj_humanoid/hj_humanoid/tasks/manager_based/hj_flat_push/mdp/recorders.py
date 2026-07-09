import torch
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
import numpy as np
import os
import glob
from typing import Optional

# Import safety value model builder (optional dependency)
try:
    from safety_value.safety_analysis_algos.model import build_mlp
    SAFETY_VALUE_MODEL_AVAILABLE = True
except ImportError:
    SAFETY_VALUE_MODEL_AVAILABLE = False
    print("[WARNING] safety_value.safety_analysis_algos.model module not found. SafetyValuePredictor will be disabled.")


class LambdaValueEnsemble(torch.nn.Module):
    """Two-critic ensemble used by λ-reachability.

    Forward returns the mean of both critics so downstream code can treat it
    like a single-valued predictor.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.critic1 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)
        self.critic2 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v1 = self.critic1(x)
        v2 = self.critic2(x)
        return 0.5 * (v1 + v2)

# ---------------------------------------------------------------------------- #
#                            Safety Signal: Balance                            #
# ---------------------------------------------------------------------------- #
@configclass
class SafetySigBalanceRecorderCfg(RecorderTermCfg):
    asset: SceneEntityCfg = SceneEntityCfg("robot")
    ball_offset: tuple[float, float, float] = (0.0, 0.0, 0.5)
    
    # z_safe: float = 0.3
    # z_optimal: float = 0.65
    phi_max: float = np.pi / 4.0 #
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="base")
    contact_threshold: float = 1.0
    support_contact_sensor_cfg: Optional[SceneEntityCfg] = SceneEntityCfg(
        "contact_forces", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
    )
    stance_clearance_min: float = 0.25  # minimum COM clearance over feet before flagging a fall
    bin_edges: tuple[float, ...] = (-0.5, 0.0, 0.5)

    key_prefix: str = "safety_signal_balance"

class SafetySigBalanceRecorder(RecorderTerm):
    def __init__(self, cfg: SafetySigBalanceRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self.robot: Articulation = env.scene[cfg.asset.name]
        self.cfg.contact_sensor_cfg.resolve(env.scene)
        if self.cfg.support_contact_sensor_cfg is not None:
            self.cfg.support_contact_sensor_cfg.resolve(env.scene)
        self.ball_offset = torch.tensor(cfg.ball_offset, device=env.device).view(1, 3).repeat(env.num_envs, 1)

        # self.z_safe = cfg.z_safe
        # self.z_optimal = cfg.z_optimal
        self.phi_max = cfg.phi_max
        self.contact_threshold = cfg.contact_threshold
        self.stance_clearance_min = cfg.stance_clearance_min
        if self.stance_clearance_min <= 0:
            raise ValueError("stance_clearance_min must be positive")

        # Optional height scanner (not all task configs include it).
        # When missing, _safety_signal_support falls back to a simple clearance heuristic
        # using the root height, so we avoid hard failures at runtime.
        self.height_scanner = self._env.scene.sensors.get("height_scanner")
        self._warned_missing_height_scanner = False

        self.bin_edges = torch.tensor(cfg.bin_edges, device=env.device, dtype=torch.float32)
    
    # def _safety_signal_z(self):
    #     root_z = self.robot.data.root_state_w[:, 2]          # (num_envs,)
    #     print("root_z:", root_z)
    #     return (self.z_safe - root_z) / (self.z_optimal - self.z_safe)

    def _safety_signal_tilt(self):
        v = self.robot.data.projected_gravity_b
        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]
        tilt = torch.atan2(torch.sqrt(x.pow(2)+y.pow(2)), -z)
        return (tilt - self.phi_max) / self.phi_max

    def _safety_signal_contact(self):
        contact_sensor: ContactSensor = self._env.scene.sensors[self.cfg.contact_sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        f = net_contact_forces[:, :, self.cfg.contact_sensor_cfg.body_ids] # (num_envs, T, body_count, 3)
        has_contact = (f.norm(dim=-1).max(dim=1)[0] > self.contact_threshold).any(dim=1)
        return has_contact.to(torch.float32) * 2.0 - 1.0  # map {False, True} -> {-1, 1}

    def _safety_signal_support(self):
        # Get root z position
        root_z = self.robot.data.root_pos_w[:, 2]  # (num_envs,)

        # Fall back gracefully when the height scanner sensor is not present in the scene.
        if self.height_scanner is None:
            if not self._warned_missing_height_scanner:
                print("[SafetySigBalanceRecorder] height_scanner not found in scene; using root height fallback for support clearance")
                self._warned_missing_height_scanner = True
            clearance = root_z  # assume terrain at z=0
            score = (self.stance_clearance_min - clearance) / self.stance_clearance_min
            return torch.clamp(score, min=-1.0, max=3.0)
        
        # Get height scanner data following the same logic as height_scan observation
        # height_scan computes: sensor_height - hit_point_z - offset
        height_scanner = self.height_scanner
        sensor_height = height_scanner.data.pos_w[:, 2]  # (num_envs,)
        scan_points_z = height_scanner.data.ray_hits_w[..., 2]  # (num_envs, num_rays)
        
        # Compute the height scan values exactly as in observations.py
        # height_scan: height = sensor_height - hit_point_z - offset
        offset = 0.5  # same as in the observation
        height_scan_values = sensor_height.unsqueeze(1) - scan_points_z - offset  # (num_envs, num_rays)
        
        # Filter out infinite values from ray misses before averaging
        # Replace inf with a large negative value so they're ignored in mean calculation
        scan_points_z_filtered = torch.where(
            torch.isfinite(scan_points_z),
            scan_points_z,
            torch.full_like(scan_points_z, float('-inf'))
        )
        
        # Average height scan (only finite values)
        # If all rays are inf, use root_z as fallback (clearance = 0)
        valid_mask = torch.isfinite(scan_points_z_filtered)
        sum_valid = (scan_points_z_filtered * valid_mask).sum(dim=-1)
        count_valid = valid_mask.sum(dim=-1).clamp(min=1)  # avoid division by zero
        avg_scan_height = sum_valid / count_valid
        
        # If no valid rays, use root_z (clearance = 0)
        avg_scan_height = torch.where(
            valid_mask.any(dim=-1),
            avg_scan_height,
            root_z
        )
        
        # Compute clearance: root_z - avg_scan_height
        clearance = root_z - avg_scan_height
        
        # Compute safety score: positive when clearance < stance_clearance_min (unsafe)
        score = (self.stance_clearance_min - clearance) / self.stance_clearance_min
        
        # Clamp score to prevent extreme values
        score = torch.clamp(score, min=-1.0, max=3.0)
        
        # Debug print
        # print(f"root_z: {root_z.mean().item():.3f}, avg_scan_height: {avg_scan_height.mean().item():.3f}, clearance: {clearance.mean().item():.3f}, score: {score.mean().item():.3f}")

        return score


    def safety_signal(self):
        l_terms: list[torch.Tensor] = []
        for name in sorted(dir(self)):
            if not name.startswith("_safety_signal_"):
                continue
            term_fn = getattr(self, name)
            if not callable(term_fn):
                continue
            term_value = term_fn()
            if term_value.ndim == 2 and term_value.shape[1] == 1:
                term_value = term_value.squeeze(-1)
            elif term_value.ndim != 1:
                raise RuntimeError(
                    f"{name} must return shape (num_envs,) or (num_envs, 1), got {tuple(term_value.shape)}"
                )
            l_terms.append(term_value)
        if not l_terms:
            raise RuntimeError("No _safety_signal_* recorders defined for aggregation.")
        stacked = torch.stack(l_terms, dim=0)
        return stacked.max(dim=0).values

    def record_post_step(self):

        l = self.safety_signal()
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        from isaaclab.utils.math import quat_apply
        ball_offset_w = quat_apply(root_quat_w, self.ball_offset)  # (num_envs, 3)
        ball_pos = root_pos_w + ball_offset_w                  # (num_envs, 3)
        bin_idx = torch.bucketize(l, self.bin_edges)  # [0..len(edges)]  shape: (num_envs,)
        bin_idx = torch.clamp(bin_idx, 0, self.bin_edges.numel()) # just in case

        if not hasattr(self._env, "safety_signal"):
            self._env.safety_signal = {}
        self._env.safety_signal["balance"] = {"pos": ball_pos, "bin": bin_idx}

        return self.cfg.key_prefix, l

    def record_post_reset(self, env_ids):
        return None, None

# ---------------------------------------------------------------------------- #
#                      Stability Signal: Velocity Tracking                     #
# ---------------------------------------------------------------------------- #
@configclass
class StabilitySigVelTrackRecorderCfg(RecorderTermCfg):
    asset: SceneEntityCfg = SceneEntityCfg("robot")
    ball_offset_tracking: tuple[float, float, float] = (0.0, 0.0, 0.5)
    ball_offset_stability: tuple[float, float, float] = (0.0, 0.0, 0.6)
    command_name: str = "base_velocity"
    bin_edges: tuple[float, ...] = (0.25, 0.5, 0.75)
    stable_steps: int = 10

    key_prefix: str = "stability_signal_vel_track"

class StabilitySigVelTrackRecorder(RecorderTerm):
    def __init__(self, cfg: StabilitySigVelTrackRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self.robot: Articulation = env.scene[cfg.asset.name]
        self.ball_offset_tracking = torch.tensor(cfg.ball_offset_tracking, device=env.device).view(1, 3).repeat(env.num_envs, 1)
        self.ball_offset_stability = torch.tensor(cfg.ball_offset_stability, device=env.device).view(1, 3).repeat(env.num_envs, 1)

        self.command_name = cfg.command_name

        def _resolve_tracking_std(term_name: str, default: float) -> float:
            reward_manager = getattr(env, "reward_manager", None)
            if reward_manager is None:
                return default
            try:
                term_cfg = reward_manager.get_term_cfg(term_name)
            except ValueError:
                return default
            return float(term_cfg.params.get("std", default))

        default_std = 0.5
        lin_std = _resolve_tracking_std("track_lin_vel_xy_exp", default_std)
        ang_std = _resolve_tracking_std("track_ang_vel_z_exp", default_std)
        self.lin_tracking_std_sq = torch.tensor(lin_std, device=env.device, dtype=torch.float32) ** 2
        self.ang_tracking_std_sq = torch.tensor(ang_std, device=env.device, dtype=torch.float32) ** 2

        # viz
        self.bin_edges = torch.tensor(cfg.bin_edges, device=env.device, dtype=torch.float32)
        self.stable_steps = int(cfg.stable_steps)
        self._tracking_streak = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    
    def record_post_step(self):

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        from isaaclab.utils.math import quat_apply

        # ----------------------------- Command tracking ----------------------------- #
        command = self._env.command_manager.get_command(self.command_name)
        lin_vel_b = self.robot.data.root_lin_vel_b[:, :2]
        ang_vel_b = self.robot.data.root_ang_vel_b[:, 2]

        lin_error = torch.sum(torch.square(command[:, :2] - lin_vel_b), dim=-1)
        lin_quality = torch.exp(-lin_error / self.lin_tracking_std_sq)

        ang_error = torch.square(command[:, 2] - ang_vel_b)
        ang_quality = torch.exp(-ang_error / self.ang_tracking_std_sq)

        tracking_quality = torch.maximum(lin_quality, ang_quality)

        ball_pos_tracking = root_pos_w + quat_apply(root_quat_w, self.ball_offset_tracking)  # (num_envs, 3)
        bin_idx_tracking = torch.bucketize(tracking_quality, self.bin_edges)
        bin_idx_tracking = torch.clamp(bin_idx_tracking, 0, self.bin_edges.numel())

        if not hasattr(self._env, "stability_signal"):
            self._env.stability_signal = {}
        self._env.stability_signal["vel_track_step"] = {"pos": ball_pos_tracking, "bin": bin_idx_tracking}

        # --------------------------------- Stability -------------------------------- #
        best_mask = bin_idx_tracking == (self.bin_edges.numel())

        if self.stable_steps <= 0:
            raise ValueError(f"stable_steps should be positive, got {self.stable_steps}")
        else:
            self._tracking_streak = torch.where(
                best_mask,
                self._tracking_streak + 1,
                torch.zeros_like(self._tracking_streak),
            )
            stable_flag = best_mask & (self._tracking_streak >= self.stable_steps)

        ball_pos_stability = root_pos_w + quat_apply(root_quat_w, self.ball_offset_stability)
        bin_idx_stability = stable_flag.to(torch.long)
        
        self._env.stability_signal["vel_track_stable"] = {"pos": ball_pos_stability, "bin": bin_idx_stability}

        return self.cfg.key_prefix, stable_flag.to(torch.float32)
    
    def record_post_reset(self, env_ids):
        if env_ids is None:
            self._tracking_streak.zero_()
        else:
            self._tracking_streak[env_ids] = 0
        return None, None

# ---------------------------------------------------------------------------- #
#                                  Event: Push                                 #
# ---------------------------------------------------------------------------- #
@configclass
class EventPushRecorderCfg(RecorderTermCfg):
    key_prefix: str = "event_push"

class EventPushRecorder(RecorderTerm):

    def __init__(self, cfg: EventPushRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

    def record_post_step(self):

        push_mask = getattr(self._env, "_event_mask_push", None)
        if push_mask is None:
            push_flag = torch.zeros(self._env.num_envs, device=self._env.device)
        else:
            push_flag = push_mask.to(torch.float32)
            push_mask.zero_()
        
        return self.cfg.key_prefix, push_flag

    def record_post_reset(self, env_ids):
        if env_ids is None:
            if hasattr(self._env, "_event_mask_push"):
                self._env._event_mask_push.zero_()
        else:
            if hasattr(self._env, "_event_mask_push"):
                self._env._event_mask_push[env_ids] = False
        return None, None

# ---------------------------------------------------------------------------- #
#                                Terminal State                                #
# ---------------------------------------------------------------------------- #
@configclass
class TerminalStateRecorderCfg(RecorderTermCfg):
    key_prefix: str = "terminal_state"

class TerminalStateRecorder(RecorderTerm):

    def __init__(self, cfg: TerminalStateRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg

    def record_post_step(self):

        reset_buf = getattr(self._env, "reset_buf", None)
        if reset_buf is None:
            terminal_flag = torch.zeros(self._env.num_envs, device=self._env.device)
        else:
            terminal_flag = reset_buf.to(torch.float32)
        
        return self.cfg.key_prefix, terminal_flag

# ---------------------------------------------------------------------------- #
#                                  Observation                                 #
# ---------------------------------------------------------------------------- #
@configclass
class ObservationRecorderCfg(RecorderTermCfg):
    key_prefix: str = "policy_obs"

class ObservationRecorder(RecorderTerm):

    def __init__(self, cfg: ObservationRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg

    def record_post_step(self):

        policy_obs = self._env.obs_buf.get("policy")
        if policy_obs is None:
            policy_obs = self._env.observation_manager.compute_group("policy")

        return self.cfg.key_prefix, policy_obs

# ---------------------------------------------------------------------------- #
#                                 Safety Value                                 #
# ---------------------------------------------------------------------------- #
@configclass
class SafetyValueInferenceCfg(RecorderTermCfg):
    """Configuration for Safety Value Inference recorder."""

    asset: SceneEntityCfg = SceneEntityCfg("robot")
    bin_edges: tuple[float, ...] = (-0.5, 0.0)
    ball_offset: tuple[float, float, float] = (0.0, 0.0, 1.2)  # offset in body frame
    
    # dataset key prefix
    key_prefix: str = "safety_value"

class SafetyValueInference(RecorderTerm):
    """Recorder that predicts safety value and visualizes it with a colored ball.
    """
    
    cfg: SafetyValueInferenceCfg
    
    def __init__(self, cfg: SafetyValueInferenceCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self.robot: Articulation = env.scene[cfg.asset.name]
        self.ball_offset = torch.tensor(cfg.ball_offset, device=env.device).view(1, 3).repeat(env.num_envs, 1)
        self.bin_edges = torch.tensor(cfg.bin_edges, device=env.device, dtype=torch.float32)
        
        # Initialize safety value model (lazy initialization - will load on first use)
        self.safety_value_model = None
        self._model_loaded = False
    
    def _load_safety_value_model(self):
        """Load the safety value model from checkpoint (lazy initialization)."""
        if self._model_loaded:
            return  # Already attempted to load
        
        self._model_loaded = True  # Mark as attempted (even if it fails)
        
        if not SAFETY_VALUE_MODEL_AVAILABLE:
            raise RuntimeError(
                "[SafetyValuePredictor] safety value model module not available. "
                "Please install the safety_value.safety_analysis_algos module to use SafetyValuePredictor."
            )
            
        if not hasattr(self._env.cfg, "safety_value_path") or self._env.cfg.safety_value_path is None:
            raise ValueError(
                "[SafetyValuePredictor] No safety_value_path provided in environment config. "
                "Please set env_cfg.safety_value_path to the directory containing the trained model."
            )

        safety_value_path = self._env.cfg.safety_value_path
        models_dir = os.path.join(safety_value_path, "models")
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"[SafetyValuePredictor] Models directory not found: {models_dir}. "
                f"Please ensure safety_value_path points to a valid training output directory."
            )
        
        # Find a checkpoint with the following priority:
        #   1) Highest step_*.pt (if any)
        #   2) Smallest-seed last_seed_<seed>.pt (deterministic choice when multiple seeds exist)
        #   3) last.pt
        step_ckpts = glob.glob(os.path.join(models_dir, "step_*.pt"))
        latest_ckpt_path = None
        latest_step = -1

        if step_ckpts:
            step_numbers = []
            for ckpt_path in step_ckpts:
                basename = os.path.basename(ckpt_path)
                try:
                    step_str = basename.split("_")[1].replace(".pt", "")
                    step_numbers.append((int(step_str), ckpt_path))
                except (IndexError, ValueError):
                    continue

            if step_numbers:
                latest_step, latest_ckpt_path = max(step_numbers, key=lambda x: x[0])

        if latest_ckpt_path is None:
            # Prefer the smallest seed among last_seed_<seed>.pt files for reproducibility.
            seed_ckpts = glob.glob(os.path.join(models_dir, "last_seed_*.pt"))
            seed_entries = []
            for ckpt_path in seed_ckpts:
                basename = os.path.basename(ckpt_path)
                try:
                    seed_str = basename.split("_")[2].replace(".pt", "")
                    seed_entries.append((int(seed_str), ckpt_path))
                except (IndexError, ValueError):
                    continue

            if seed_entries:
                seed_entries.sort(key=lambda x: x[0])  # smallest seed first
                chosen_seed, latest_ckpt_path = seed_entries[0]
                latest_step = -1  # unknown
                print(
                    f"[SafetyValuePredictor] Using smallest-seed checkpoint: {latest_ckpt_path} (seed {chosen_seed})"
                )

        if latest_ckpt_path is None:
            last_checkpoint = os.path.join(models_dir, "last.pt")
            if os.path.exists(last_checkpoint):
                latest_ckpt_path = last_checkpoint
                temp_ckpt = torch.load(last_checkpoint, map_location='cpu')
                latest_step = temp_ckpt.get('global_step', -1)
            else:
                raise FileNotFoundError(
                    f"[SafetyValuePredictor] No checkpoints found in {models_dir}. "
                    f"Please ensure the model has been trained and checkpoints exist."
                )
        
        print(f"[SafetyValuePredictor] Loading safety value model from: {latest_ckpt_path} (step {latest_step})")
        
        # Load checkpoint
        ckpt = torch.load(latest_ckpt_path, map_location=self._env.device)
        
        # Get input dimension from policy observation space
        input_dim = self._env.observation_manager.group_obs_dim["policy"][0]
        
        # Read hidden dimensions and algorithm from training config
        config_path = os.path.join(safety_value_path, "training_config.json")
        if not os.path.exists(config_path):
            print(
                f"[SafetyValuePredictor] WARNING: training_config.json not found at {config_path}, "
                "using default hidden_dims=(256, 256) and assuming single-critic"
            )
            hidden_dims = (256, 256)
            algorithm = "dpe"
        else:
            import json
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            hidden_dims = tuple(training_config.get("hidden_dims", [256, 256]))
            algorithm = training_config.get("algorithm", "dpe")
            print(
                f"[SafetyValuePredictor] Loaded hidden_dims={hidden_dims}, algorithm={algorithm} "
                "from training_config.json"
            )

        # Build model with same architecture as training
        if algorithm == "lambda_reachability":
            self.safety_value_model = LambdaValueEnsemble(input_dim=input_dim, hidden_dims=hidden_dims).to(
                self._env.device
            )
            try:
                self.safety_value_model.critic1.load_state_dict(ckpt["critic1"])
                self.safety_value_model.critic2.load_state_dict(ckpt["critic2"])
            except KeyError as exc:
                raise KeyError(
                    "[SafetyValuePredictor] λ-reachability checkpoint missing 'critic1'/'critic2' states"
                ) from exc
        else:
            self.safety_value_model = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self._env.device)
            try:
                self.safety_value_model.load_state_dict(ckpt["model_state"])
            except KeyError as exc:
                raise KeyError(
                    "[SafetyValuePredictor] checkpoint missing 'model_state'; expected single-critic checkpoint"
                ) from exc

        self.safety_value_model.eval()
        
        print(
            f"[SafetyValuePredictor] Successfully loaded safety value model with input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}"
        )
    
    def record_post_step(self):
        """Predict HJ value and update visualization."""
        # Lazy load model on first use
        if not self._model_loaded:
            self._load_safety_value_model()
        
        if self.safety_value_model is None:
            raise RuntimeError(
                "[SafetyValuePredictor] safety value model failed to load. "
                "Please check the error messages above for details."
            )
        
        # Get policy observations
        policy_obs = self._env.observation_manager.compute_group("policy")
        
        # Predict HJ values
        with torch.no_grad():
            safety_values = self.safety_value_model(policy_obs)  # (num_envs, 1)
        
        # Store predictions (squeeze to 1D, but keep at least 1D)
        if safety_values.shape[-1] == 1:
            safety_values = safety_values.squeeze(-1)
        
        # Ensure we always have at least 1D tensor (handle single env case)
        if safety_values.ndim == 0:
            safety_values = safety_values.unsqueeze(0)
        
        # Update ball positions (behind robot in body frame)
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        
        # Transform body frame offset to world frame
        from isaaclab.utils.math import quat_apply
        ball_offset_w = quat_apply(root_quat_w, self.ball_offset)
        
        ball_pos = root_pos_w + ball_offset_w
        
        # Update bin index (0 = green/safe, 1 = red/unsafe)
        bin_idx = torch.bucketize(safety_values, self.bin_edges)
        bin_idx = torch.clamp(bin_idx, 0, self.bin_edges.numel())
        
        # Store in environment for event access
        if not hasattr(self._env, "safety_value"):
            self._env.safety_value = {}
        self._env.safety_value["prediction"] = {"pos": ball_pos, "bin": bin_idx}
        
        return self.cfg.key_prefix, safety_values.to(torch.float32)

