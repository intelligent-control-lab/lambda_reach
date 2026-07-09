# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms for collision avoidance task - safety signal computation."""

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
#                        Safety Signal: Collision Avoidance                    #
# ---------------------------------------------------------------------------- #
@configclass
class SafetySigCollisionRecorderCfg(RecorderTermCfg):
    """Configuration for collision avoidance safety signal recorder.
    
    Safety signal is computed as the max of multiple components:
    - Collision: ball contact with robot body
    - Tilt: robot tilting beyond safe angle
    - Body contact: torso/base touching ground
    - Support: robot height clearance above terrain
    
    This ensures the robot must avoid ball collisions AND maintain balance.
    """
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball")
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("ball_contact")
    
    # Visualization ball offset above robot (for safety signal display)
    ball_offset: tuple[float, float, float] = (0.0, 0.0, 0.8)
    
    # -------------------- Collision parameters -------------------- #
    # Distance at which safety signal approaches 0 (but stays negative without contact)
    danger_distance: float = 0.5
    
    # Contact force threshold for ball contact
    ball_contact_threshold: float = 0.1
    
    # Latch duration: how long to hold the unsafe state after contact (seconds)
    # This makes the red color visible for longer after a hit
    latch_duration: float = 0.25
    
    # -------------------- Balance parameters (from flat_push) -------------------- #
    # Maximum tilt angle before considered unsafe
    phi_max: float = np.pi / 4.0
    
    # Body contact sensor (for detecting torso/base hitting ground)
    body_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="torso_link")
    body_contact_threshold: float = 1.0
    
    # Minimum COM clearance over terrain before flagging a fall
    stance_clearance_min: float = 0.25
    
    # Bin edges for visualization: (-0.5, 0.0) => 3 bins (green/yellow/red)
    bin_edges: tuple[float, ...] = (-0.5, 0.0)
    
    key_prefix: str = "safety_signal_collision"


class SafetySigCollisionRecorder(RecorderTerm):
    """Recorder that computes safety signal based on ball collision AND balance.
    
    The safety signal l is the max of multiple components:
    - Collision: ball contact with robot body (with latching)
    - Tilt: robot tilting beyond safe angle
    - Body contact: torso/base touching ground
    - Support: robot height clearance above terrain
    
    This ensures the robot must avoid ball collisions AND maintain balance.
    """
    
    def __init__(self, cfg: SafetySigCollisionRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        
        # Get robot and ball assets
        self.robot: Articulation = env.scene[cfg.robot_cfg.name]
        self.ball: RigidObject = env.scene[cfg.ball_cfg.name]
        
        # Get ball contact sensor
        self.ball_contact_sensor: ContactSensor = env.scene.sensors[cfg.contact_sensor_cfg.name]
        
        # Precompute ball offset tensor
        self.ball_offset = torch.tensor(
            cfg.ball_offset, device=env.device
        ).view(1, 3).repeat(env.num_envs, 1)
        
        # -------------------- Collision parameters -------------------- #
        self.danger_distance = cfg.danger_distance
        if self.danger_distance <= 0:
            raise ValueError("danger_distance must be positive")
        self.ball_contact_threshold = cfg.ball_contact_threshold
        
        # Latch state: tracks when each env last had a collision
        # -inf means no collision has occurred yet
        self.latch_duration = cfg.latch_duration
        self._last_collision_time = torch.full(
            (env.num_envs,), float('-inf'), device=env.device, dtype=torch.float32
        )
        
        # -------------------- Balance parameters -------------------- #
        self.phi_max = cfg.phi_max
        self.body_contact_threshold = cfg.body_contact_threshold
        self.stance_clearance_min = cfg.stance_clearance_min
        if self.stance_clearance_min <= 0:
            raise ValueError("stance_clearance_min must be positive")
        
        # Resolve body contact sensor config
        self.cfg.body_contact_sensor_cfg.resolve(env.scene)
        
        # Optional height scanner (not all task configs include it)
        self.height_scanner = self._env.scene.sensors.get("height_scanner")
        self._warned_missing_height_scanner = False
        
        # Bin edges for visualization
        self.bin_edges = torch.tensor(cfg.bin_edges, device=env.device, dtype=torch.float32)
    
    def _detect_ball_contact(self) -> torch.Tensor:
        """Detect if ball is in contact with robot (any part).
        
        Since the contact sensor is on the ball, any contact detected means
        the ball hit something. We filter out ground contacts by checking
        if the ball is above a minimum height.
        
        Returns:
            Boolean tensor of shape (num_envs,) - True if robot contact detected
        """
        # Get net contact forces on ball
        net_forces = self.ball_contact_sensor.data.net_forces_w_history  # (num_envs, T, num_bodies, 3)
        forces_magnitude = net_forces.norm(dim=-1)  # (num_envs, T, num_bodies)
        max_force = forces_magnitude.max(dim=1)[0].max(dim=1)[0]  # (num_envs,)
        
        # Detect which envs have contact
        has_contact = max_force > self.ball_contact_threshold
        
        # Get ball height - filter out ground contacts (ball at ground level)
        # Ball radius is 0.1m, so when touching ground, center is at 0.1m
        # Use 0.15m threshold to safely filter out ground contacts
        ball_height = self.ball.data.root_pos_w[:, 2]  # (num_envs,)
        is_above_ground = ball_height > 0.15  # Must be higher than ball radius
        
        # Contact with robot = contact AND ball is above ground
        return has_contact & is_above_ground
    
    def _safety_signal_collision(self) -> torch.Tensor:
        """Compute safety signal based on contact and distance to ball.
        
        Returns:
            Safety signal l: shape (num_envs,)
            - Contact: l = 1.0
            - No contact: l = (danger_distance - distance) / danger_distance, clamped to [-1, 0)
        """
        # Check if ball is enabled via curriculum flag
        ball_enabled = getattr(self._env, "_ball_enabled", True)
        
        if not ball_enabled:
            # Ball is disabled - return safe value
            return torch.full(
                (self._env.num_envs,), -1.0, 
                device=self._env.device, dtype=torch.float32
            )
        
        # Check for body contact
        has_ball_contact = self._detect_ball_contact()
        
        # Get positions
        robot_pos = self.robot.data.root_pos_w  # (num_envs, 3)
        ball_pos = self.ball.data.root_pos_w    # (num_envs, 3)
        
        # Compute 3D distance
        distance = torch.norm(ball_pos - robot_pos, dim=-1)  # (num_envs,)
        
        # Compute distance-based safety signal (for no-contact case)
        # This gives values in [-1, +inf) based on distance
        l_distance = (self.danger_distance - distance) / self.danger_distance
        
        # Clamp to [-1, 0) for no-contact case (always negative = safe)
        l_no_contact = torch.clamp(l_distance, min=-1.0, max=-1e-6)
        
        # Update latch: record current time for envs with new collision
        current_time = self._env.episode_length_buf * self._env.step_dt  # Current sim time per env
        self._last_collision_time = torch.where(
            has_ball_contact,
            current_time,
            self._last_collision_time
        )
        
        # Check if we're within latch duration of last collision
        time_since_collision = current_time - self._last_collision_time
        is_latched = time_since_collision < self.latch_duration
        
        # Final safety signal: 1.0 if contact OR latched, otherwise distance-based (negative)
        l = torch.where(has_ball_contact | is_latched, torch.ones_like(l_no_contact), l_no_contact)
        
        return l
    
    def _safety_signal_tilt(self) -> torch.Tensor:
        """Compute safety signal based on robot tilt angle.
        
        Returns:
            Safety signal l: shape (num_envs,)
            - l > 0 when tilt exceeds phi_max (unsafe)
            - l < 0 when tilt is within bounds (safe)
        """
        v = self.robot.data.projected_gravity_b
        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]
        tilt = torch.atan2(torch.sqrt(x.pow(2) + y.pow(2)), -z)
        return (tilt - self.phi_max) / self.phi_max
    
    def _safety_signal_contact(self) -> torch.Tensor:
        """Compute safety signal based on body (torso) contact with ground.
        
        Returns:
            Safety signal l: shape (num_envs,)
            - l = 1.0 when body contact detected (unsafe)
            - l = -1.0 when no contact (safe)
        """
        body_contact_sensor: ContactSensor = self._env.scene.sensors[self.cfg.body_contact_sensor_cfg.name]
        net_contact_forces = body_contact_sensor.data.net_forces_w_history
        f = net_contact_forces[:, :, self.cfg.body_contact_sensor_cfg.body_ids]  # (num_envs, T, body_count, 3)
        has_contact = (f.norm(dim=-1).max(dim=1)[0] > self.body_contact_threshold).any(dim=1)
        return has_contact.to(torch.float32) * 2.0 - 1.0  # map {False, True} -> {-1, 1}
    
    def _safety_signal_support(self) -> torch.Tensor:
        """Compute safety signal based on robot height clearance above terrain.
        
        Returns:
            Safety signal l: shape (num_envs,)
            - l > 0 when clearance < stance_clearance_min (unsafe)
            - l < 0 when clearance is sufficient (safe)
        """
        # Get root z position
        root_z = self.robot.data.root_pos_w[:, 2]  # (num_envs,)
        
        # Fall back gracefully when the height scanner sensor is not present in the scene
        if self.height_scanner is None:
            if not self._warned_missing_height_scanner:
                print("[SafetySigCollisionRecorder] height_scanner not found in scene; using root height fallback for support clearance")
                self._warned_missing_height_scanner = True
            clearance = root_z  # assume terrain at z=0
            score = (self.stance_clearance_min - clearance) / self.stance_clearance_min
            return torch.clamp(score, min=-1.0, max=3.0)
        
        # Get height scanner data
        height_scanner = self.height_scanner
        sensor_height = height_scanner.data.pos_w[:, 2]  # (num_envs,)
        scan_points_z = height_scanner.data.ray_hits_w[..., 2]  # (num_envs, num_rays)
        
        # Filter out infinite values from ray misses
        scan_points_z_filtered = torch.where(
            torch.isfinite(scan_points_z),
            scan_points_z,
            torch.full_like(scan_points_z, float('-inf'))
        )
        
        # Average height scan (only finite values)
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
        return torch.clamp(score, min=-1.0, max=3.0)
    
    def safety_signal(self) -> torch.Tensor:
        """Aggregate all safety signal terms (max over all).
        
        Combines collision avoidance with balance signals:
        - collision: ball contact with robot body
        - tilt: robot tilting beyond safe angle
        - contact: torso/base touching ground
        - support: robot height clearance above terrain
        """
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
        """Record safety signal and update visualization data."""
        l = self.safety_signal()
        
        # Compute visualization ball position (above robot)
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        from isaaclab.utils.math import quat_apply
        ball_offset_w = quat_apply(root_quat_w, self.ball_offset)  # (num_envs, 3)
        ball_pos = root_pos_w + ball_offset_w  # (num_envs, 3)
        
        # Compute bin index for visualization
        bin_idx = torch.bucketize(l, self.bin_edges)  # [0..len(edges)]
        bin_idx = torch.clamp(bin_idx, 0, self.bin_edges.numel())
        
        # Store in env for visualization event to pick up
        if not hasattr(self._env, "safety_signal"):
            self._env.safety_signal = {}
        self._env.safety_signal["collision"] = {"pos": ball_pos, "bin": bin_idx}
        
        return self.cfg.key_prefix, l
    
    def record_post_reset(self, env_ids):
        """Reset latch state for reset environments."""
        if env_ids is not None and len(env_ids) > 0:
            self._last_collision_time[env_ids] = float('-inf')
        return None, None


# ---------------------------------------------------------------------------- #
#                        Event: Ball Spawn (for segmenting)                    #
# ---------------------------------------------------------------------------- #
@configclass
class EventBallSpawnRecorderCfg(RecorderTermCfg):
    """Configuration for recording ball spawn events.
    
    Ball spawns are segmenting events - they mark points where
    the future trajectory becomes unpredictable, so HJ data should
    be segmented at these points.
    """
    key_prefix: str = "event_ball_spawn"


class EventBallSpawnRecorder(RecorderTerm):
    """Recorder that tracks when balls are spawned toward the robot.
    
    The spawn event is set by the spawn_ball_towards_robot event function
    via env._event_mask_ball_spawn flag.
    """
    
    def __init__(self, cfg: EventBallSpawnRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
    
    def record_post_step(self):
        """Record whether a ball spawn occurred this step."""
        spawn_mask = getattr(self._env, "_event_mask_ball_spawn", None)
        if spawn_mask is None:
            spawn_flag = torch.zeros(self._env.num_envs, device=self._env.device)
        else:
            spawn_flag = spawn_mask.to(torch.float32)
            spawn_mask.zero_()  # Reset the flag
        
        return self.cfg.key_prefix, spawn_flag
    
    def record_post_reset(self, env_ids):
        """Clear spawn flag on reset."""
        if env_ids is None:
            if hasattr(self._env, "_event_mask_ball_spawn"):
                self._env._event_mask_ball_spawn.zero_()
        else:
            if hasattr(self._env, "_event_mask_ball_spawn"):
                self._env._event_mask_ball_spawn[env_ids] = False
        return None, None


# ---------------------------------------------------------------------------- #
#                            Terminal State                                    #
# ---------------------------------------------------------------------------- #
@configclass
class TerminalStateRecorderCfg(RecorderTermCfg):
    key_prefix: str = "terminal_state"


class TerminalStateRecorder(RecorderTerm):
    """Records terminal state flag from environment termination manager."""
    
    def __init__(self, cfg: TerminalStateRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
    
    def record_post_step(self):
        """Record if environment terminated this step."""
        terminated = self._env.termination_manager.terminated
        return self.cfg.key_prefix, terminated.to(torch.float32)
    
    def record_post_reset(self, env_ids):
        return None, None


# ---------------------------------------------------------------------------- #
#                           Observation Recorder                               #
# ---------------------------------------------------------------------------- #
@configclass
class ObservationRecorderCfg(RecorderTermCfg):
    key_prefix: str = "policy_obs"


class ObservationRecorder(RecorderTerm):
    """Records policy observations for trajectory logging."""
    
    def __init__(self, cfg: ObservationRecorderCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
    
    def record_post_step(self):
        """Record the current policy observations."""
        obs = self._env.observation_manager.compute_group("policy")
        return self.cfg.key_prefix, obs
    
    def record_post_reset(self, env_ids):
        return None, None


# ---------------------------------------------------------------------------- #
#                        Safety Value Inference                                #
# ---------------------------------------------------------------------------- #
@configclass
class SafetyValueInferenceCfg(RecorderTermCfg):
    """Configuration for safety value inference during rollouts."""
    asset: SceneEntityCfg = SceneEntityCfg("robot")
    ball_offset: tuple[float, float, float] = (0.0, 0.0, 1.2)
    bin_edges: tuple[float, ...] = (-0.5, 0.0)
    key_prefix: str = "safety_value"


class SafetyValueInference(RecorderTerm):
    """Recorder that predicts safety value and visualizes it with a colored ball.
    
    Loads a trained safety value model and runs it on current observations
    to predict safety value. Used for comparing learned vs ground truth.
    """
    
    cfg: SafetyValueInferenceCfg
    
    def __init__(self, cfg: SafetyValueInferenceCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        
        self.robot: Articulation = env.scene[cfg.asset.name]
        self.ball_offset = torch.tensor(
            cfg.ball_offset, device=env.device
        ).view(1, 3).repeat(env.num_envs, 1)
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
        """Predict safety value and update visualization."""
        # Lazy load model on first use
        if not self._model_loaded:
            self._load_safety_value_model()
        
        if self.safety_value_model is None:
            raise RuntimeError(
                "[SafetyValuePredictor] safety value model failed to load. "
                "Please check the error messages above for details."
            )
        
        # Get policy observations
        obs = self._env.observation_manager.compute_group("policy")
        
        # Predict safety values
        with torch.no_grad():
            safety_values = self.safety_value_model(obs)  # (num_envs, 1)
        
        # Store predictions (squeeze to 1D, but keep at least 1D)
        if safety_values.shape[-1] == 1:
            safety_values = safety_values.squeeze(-1)
        
        # Ensure we always have at least 1D tensor (handle single env case)
        if safety_values.ndim == 0:
            safety_values = safety_values.unsqueeze(0)
        
        # Compute visualization position
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        from isaaclab.utils.math import quat_apply
        ball_offset_w = quat_apply(root_quat_w, self.ball_offset)
        ball_pos = root_pos_w + ball_offset_w
        
        # Compute bin index
        bin_idx = torch.bucketize(safety_values, self.bin_edges)
        bin_idx = torch.clamp(bin_idx, 0, self.bin_edges.numel())
        
        # Store for visualization
        if not hasattr(self._env, "safety_value"):
            self._env.safety_value = {}
        self._env.safety_value["prediction"] = {"pos": ball_pos, "bin": bin_idx}
        
        return self.cfg.key_prefix, safety_values.to(torch.float32)
    
    def record_post_reset(self, env_ids):
        return None, None
