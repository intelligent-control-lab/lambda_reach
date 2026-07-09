"""
Inference script for safety value visualization and validation.

This script runs a trained policy with real-time safety value prediction and visualization.
It loads a trained safety value model and displays predictions alongside the policy rollout.

Usage:
    python safety_value/scripts/safety_analysis/inference.py \\
        --task Isaac-Velocity-Flat-G1 \\
        --safety_analysis_root exp_name \\
        --model_subdir discounted_policy_evaluation_lam0.9 \\
        --checkpoint path/to/policy.pt \\
        --video \\
        --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

# Base directory for all safety analysis data and results
SAFETY_ANALYSIS_BASE = "logs/safety_analysis"
CAMERA_DISTANCE_FACTOR = 1.0 / 3.0
MANUAL_CAMERA_BASE_OFFSET = [3.5, 3.5, 2.5]

# add argparse arguments
parser = argparse.ArgumentParser(description="Run policy with safety value inference and visualization.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during inference.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint file.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory where recorder datasets and videos should be stored during inference.")
parser.add_argument("--dataset_name", type=str, default=None, help="Dataset filename (without extension) to use for recorded episodes during inference.")
parser.add_argument("--safety_analysis_root", type=str, default=None, help='Safety analysis experiment name (e.g., "g1_flat_ppo_1000")')
parser.add_argument("--model_subdir", type=str, default=None, help='Model subdirectory name (e.g., "discounted_policy_evaluation_lam0.9")')
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


def build_output_directory(
    safety_analysis_root: str,
    model_subdir: str,
    seed: int,
    custom_output_dir: str = None
) -> Path:
    """Build output directory path.
    
    Args:
        safety_analysis_root: Safety analysis experiment name
        model_subdir: Model subdirectory name
        seed: Random seed
        custom_output_dir: Optional custom output directory
        
    Returns:
        Absolute path to output directory
    """
    if custom_output_dir is not None:
        output_dir = Path(custom_output_dir).resolve()
    else:
        # Default structure: logs/safety_analysis/<root>/results/<model_subdir>/inference/seed_<seed>
        output_dir = Path(SAFETY_ANALYSIS_BASE) / safety_analysis_root / "results" / model_subdir / "inference" / f"seed_{seed}"
        output_dir = output_dir.resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_safety_value_path(safety_analysis_root: str, model_subdir: str) -> Path:
    """Get path to trained safety value model.
    
    Args:
        safety_analysis_root: Safety analysis experiment name
        model_subdir: Model subdirectory name
        
    Returns:
        Absolute path to safety value model directory
    """
    model_dir = Path(SAFETY_ANALYSIS_BASE) / safety_analysis_root / "results" / model_subdir
    model_dir = model_dir.resolve()
    
    # Verify that the model directory exists
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Safety value model directory not found: {model_dir}")

    return model_dir.resolve()


def infer_tracked_asset_name(env: Any) -> Optional[str]:
    """Pick a sensible default asset to track for single-env visualizations."""

    scene = getattr(env, "scene", None)
    if scene is None:
        return None

    articulations = list(getattr(scene, "articulations", {}).keys())
    if "robot" in articulations:
        return "robot"
    if articulations:
        return articulations[0]

    rigid_objects = list(getattr(scene, "rigid_objects", {}).keys())
    return rigid_objects[0] if rigid_objects else None


def configure_viewport_camera_tracking(env: Any, asset_name: Optional[str]) -> tuple[bool, Optional[Any]]:
    """Try to enable Isaac Lab's viewport camera controller tracking."""

    if asset_name is None:
        return False, None

    viewport_controller = getattr(env, "viewport_camera_controller", None)
    if viewport_controller is None:
        return False, None

    try:
        viewport_controller.set_view_env_index(0)
        shrink_viewport_camera_distance(viewport_controller, CAMERA_DISTANCE_FACTOR)
        viewport_controller.update_view_to_asset_root(asset_name)
        print(f"[INFO] Camera tracking enabled via viewport controller (asset='{asset_name}')")
        return True, viewport_controller
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(f"[WARNING] Failed to configure viewport controller tracking: {exc}")
        return False, None


def teardown_viewport_camera_tracking(viewport_controller: Optional[Any]):
    """Reset viewport tracking to avoid callbacks firing after the env shuts down."""

    if viewport_controller is None:
        return

    try:
        viewport_controller.update_view_to_world()
        viewport_controller.cfg.asset_name = None
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(f"[WARNING] Failed to tear down viewport controller tracking cleanly: {exc}")


def setup_manual_camera_helper(env: Any, simulation_app: Any, asset_name: Optional[str]):
    """Configure manual camera tracking helpers for cases where the viewport controller isn't available."""

    if asset_name is None:
        return None

    try:
        import omni.kit.viewport.utility
        from pxr import Gf

        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api:
            print("[INFO] Camera tracking enabled via viewport API fallback")
            return {"viewport": viewport_api, "Gf": Gf, "method": "viewport"}
    except Exception as exc:
        try:
            print(f"[INFO] Viewport API fallback unavailable ({exc}), trying simulation camera...")
            return {"method": "sim_camera", "sim_app": simulation_app}
        except Exception as fallback_exc:
            print(f"[WARNING] Could not setup camera tracking fallbacks: {fallback_exc}")

    return None


def shrink_viewport_camera_distance(viewport_controller: Any, scale: float):
    """Move the viewport controller eye closer to the tracked asset while keeping view direction."""

    if viewport_controller is None or scale <= 0:
        return

    try:
        eye = np.asarray(viewport_controller.default_cam_eye, dtype=float)
        lookat = np.asarray(viewport_controller.default_cam_lookat, dtype=float)
        delta = eye - lookat
        viewport_controller.default_cam_eye = lookat + delta * scale
        viewport_controller.update_view_location()
        print(f"[INFO] Viewport controller camera distance scaled by {scale:.3f}")
    except Exception as exc:
        print(f"[WARNING] Could not scale viewport camera distance: {exc}")


def update_manual_camera(helper: Optional[dict], env: Any, asset_name: Optional[str]):
    """Update manual camera helpers to follow the tracked asset."""

    if helper is None or asset_name is None:
        return

    scene = getattr(env, "scene", None)
    if scene is None or asset_name not in scene:
        return

    try:
        tracked_entity = scene[asset_name]
        robot_pos = tracked_entity.data.root_pos_w[0].cpu().numpy()
        cam_offset = [offset * CAMERA_DISTANCE_FACTOR for offset in MANUAL_CAMERA_BASE_OFFSET]

        if helper.get("method") == "viewport":
            gf = helper["Gf"]
            eye = gf.Vec3d(*(float(robot_pos[i] + cam_offset[i]) for i in range(3)))
            target = gf.Vec3d(*(float(robot_pos[i]) for i in range(3)))
            helper["viewport"].set_camera_position("/OmniverseKit_Persp", eye, target)
        elif helper.get("method") == "sim_camera":
            import carb

            settings = carb.settings.get_settings()
            eye = [float(robot_pos[i] + cam_offset[i]) for i in range(3)]
            target = [float(robot_pos[i]) for i in range(3)]
            settings.set("/app/viewport/camPos", eye)
            settings.set("/app/viewport/camTarget", target)
    except Exception as exc:
        if not helper.get("_error_printed"):
            print(f"[WARNING] Camera tracking error: {exc}")
            helper["_error_printed"] = True


import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import hj_humanoid.tasks  # noqa: F401


@hydra_task_config(args_cli.task + "-SAFETY-INFERENCE", "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run inference with safety value prediction."""
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    
    # Validate required arguments
    if not args_cli.safety_analysis_root:
        raise ValueError("--safety_analysis_root is required")
    if not args_cli.model_subdir:
        raise ValueError("--model_subdir is required")
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required")
    
    # Append SAFETY-INFERENCE to task name if not present
    task_name = args_cli.task
    if not task_name.endswith("-SAFETY-INFERENCE"):
        task_name = f"{task_name}-SAFETY-INFERENCE"
    
    # Use seed from args if provided, otherwise use agent config default
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    
    # Build output directory
    output_dir = build_output_directory(
        args_cli.safety_analysis_root,
        args_cli.model_subdir,
        seed,
        args_cli.output_dir
    )
    
    # Get safety value model path
    safety_value_path = get_safety_value_path(args_cli.safety_analysis_root, args_cli.model_subdir)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Safety Value Inference Configuration:")
    print("=" * 80)
    print(f"Task: {args_cli.task} → {task_name}")
    print(f"Policy checkpoint: {args_cli.checkpoint}")
    print(f"Safety analysis root: {args_cli.safety_analysis_root}")
    print(f"Safety value model: {args_cli.model_subdir}")
    print(f"Seed: {seed}")
    print(f"Number of environments: {args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs}")
    print(f"Output directory: {output_dir}")
    print(f"Safety value path: {safety_value_path}")
    if args_cli.video:
        print(f"Video recording: Enabled (length={args_cli.video_length})")
    print("=" * 80 + "\n")
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Set output directory
    env_cfg.log_dir = str(output_dir)
    
    # Configure dataset export if recorders are available
    if getattr(env_cfg, "recorders", None) is not None:
        env_cfg.recorders.dataset_export_dir_path = str(output_dir)
        if args_cli.dataset_name:
            env_cfg.recorders.dataset_filename = args_cli.dataset_name
    
    # Pass safety value path to environment config
    env_cfg.safety_value_path = str(safety_value_path)
    print(f"[INFO] Safety value path set in env config: {env_cfg.safety_value_path}")
    
    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    tracked_asset_name: str | None = None
    camera_controller_in_use = False
    viewport_controller = None
    camera_helper = None

    if args_cli.num_envs == 1:
        tracked_asset_name = infer_tracked_asset_name(base_env)
        if tracked_asset_name is None:
            print("[WARNING] No articulations or rigid objects found to track")
        else:
            camera_controller_in_use, viewport_controller = configure_viewport_camera_tracking(
                base_env, tracked_asset_name
            )

        if not args_cli.video and not camera_controller_in_use:
            camera_helper = setup_manual_camera_helper(base_env, simulation_app, tracked_asset_name)
    
    # Wrap for video recording
    if args_cli.video:
        video_folder = output_dir / "videos" / "inference"
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_folder}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load checkpoint
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    
    # Create runner and load policy
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Run inference
    print("\n" + "=" * 80)
    print("Starting inference...")
    print("=" * 80 + "\n")
    
    obs = env.get_observations()
    timestep = 0
    episode_count = 0
    
    # Track resets for single environment mode
    if args_cli.num_envs == 1:
        print("[INFO] Single environment mode: Simulation will end after first episode completion")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, terminated, truncated = env.step(actions)
            
            update_manual_camera(camera_helper, base_env, tracked_asset_name)
            
            # Check for episode termination in single environment mode
            if args_cli.num_envs == 1:
                reset_buf = getattr(base_env, "reset_buf", None)
                if reset_buf is not None and reset_buf[0].item():
                    episode_count += 1
                    print(f"[INFO] Episode {episode_count} completed. Ending simulation.")
                    break

            if args_cli.video:
                timestep += 1
                if timestep == 1 or timestep % 50 == 0:
                    print(f"[PROGRESS] Recorded {timestep}/{args_cli.video_length} frames")
                if timestep == args_cli.video_length:
                    break
    finally:
        if camera_controller_in_use:
            teardown_viewport_camera_tracking(viewport_controller)
    
    # Cleanup
    env.close()
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
