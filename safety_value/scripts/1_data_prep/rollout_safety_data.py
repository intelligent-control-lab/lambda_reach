#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to rollout a trained RL policy to collect safety data for HJ value training."""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Rollout RL policy to collect safety data for HJ value training.")
parser.add_argument("--task", type=str, required=True, help="Base task name (e.g., Isaac-Velocity-Flat-G1-PPO)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during rollout")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps)")
parser.add_argument("--dataset_name", type=str, default="data_raw", help="Dataset filename (without extension)")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--safety_analysis_root",
    type=str,
    default=None,
    help=(
        "Optional output experiment name under logs/safety_analysis. "
        "Defaults to <policy_experiment>_<checkpoint_step> inferred from the checkpoint path."
    ),
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import hj_humanoid.tasks  # noqa: F401


def extract_exp_info(checkpoint_path: str) -> tuple[str, int]:
    """Extract experiment name and step number from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., .../logs/rsl_rl/g1_flat_ppo/2025-11-27_22-28-58/model_1000.pt)
        
    Returns:
        Tuple of (experiment_name, step_number)
        
    Example:
        >>> extract_exp_info("/home/user/logs/rsl_rl/g1_flat_ppo/2025-11-27_22-28-58/model_1000.pt")
        ('g1_flat_ppo', 1000)
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Extract step number from filename (e.g., model_1000.pt -> 1000)
    filename = os.path.basename(checkpoint_path)
    step_match = re.search(r'model_(\d+)\.pt', filename)
    if not step_match:
        raise ValueError(f"Could not extract step number from checkpoint filename: {filename}")
    step = int(step_match.group(1))
    
    # Extract experiment name from path (directory under rsl_rl/)
    # Path structure: .../logs/rsl_rl/<exp_name>/<timestamp>/model_*.pt
    path_parts = Path(checkpoint_path).parts
    try:
        rsl_rl_idx = path_parts.index("rsl_rl")
        exp_name = path_parts[rsl_rl_idx + 1]
    except (ValueError, IndexError):
        raise ValueError(f"Could not extract experiment name from checkpoint path: {checkpoint_path}")
    
    return exp_name, step


@hydra_task_config(args_cli.task + "-SAFETY-ROLLOUT", args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Rollout trained policy to collect safety data."""
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    # Resolve output experiment name.
    if args_cli.safety_analysis_root is None:
        exp_name, step = extract_exp_info(args_cli.checkpoint)
        safety_analysis_root = f"{exp_name}_{step}"
        print(f"[INFO] Experiment: {exp_name}, Step: {step}")
    else:
        safety_analysis_root = args_cli.safety_analysis_root
        print(f"[INFO] Safety analysis root: {safety_analysis_root}")

    output_dir = os.path.abspath(f"./logs/safety_analysis/{safety_analysis_root}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # Configure recorder
    if getattr(env_cfg, "recorders", None) is not None:
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = args_cli.dataset_name
    
    env_cfg.log_dir = output_dir
    
    # Load checkpoint
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    
    # Create environment
    task_name = args_cli.task + "-SAFETY-ROLLOUT"
    print(f"[INFO] Creating environment: {task_name}")
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Setup video recording
    if args_cli.video:
        video_folder = os.path.join(output_dir, "videos", "rollout")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos. Saving to: {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load trained policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    
    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    dt = env.unwrapped.step_dt
    
    # Reset environment
    obs = env.get_observations()
    timestep = 0
    
    # Rollout loop
    print(f"[INFO] Starting rollout for {args_cli.video_length} steps...")
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run policy inference
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if timestep == 1 or timestep % 200 == 0:
            label = "Recorded" if args_cli.video else "Collected"
            print(f"[PROGRESS] {label} {timestep}/{args_cli.video_length} steps")

        if timestep == args_cli.video_length:
            break
    
    # Close environment
    env.close()
    print(f"[INFO] Rollout complete. Data saved to: {output_dir}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
