#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained RL policy."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL policy.")
parser.add_argument("--task", type=str, required=True, help="Base task name (e.g., Isaac-Velocity-Flat-G1-PPO)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps)")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible")
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

import gymnasium as gym
import importlib.metadata as metadata
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import hj_humanoid.tasks  # noqa: F401


class _PolicyExportWrapper(torch.nn.Module):
    """Adapts newer rsl-rl actor models to Isaac Lab's legacy policy exporter API."""

    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        self.actor = actor
        self.is_recurrent = getattr(actor, "is_recurrent", False)


@hydra_task_config(args_cli.task + "-POLICY", args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with trained RL policy."""
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    
    # Load checkpoint
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    
    # Set log directory (same directory as checkpoint)
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir
    
    # Create environment
    task_name = args_cli.task + "-POLICY"
    print(f"[INFO] Creating environment: {task_name}")
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Setup video recording
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
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
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint_path)
    
    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Export policy to onnx/jit
    try:
        export_model_dir = os.path.join(log_dir, "exported")
        if hasattr(runner, "export_policy_to_jit") and hasattr(runner, "export_policy_to_onnx"):
            runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
        else:
            # Extract the neural network module for older rsl-rl versions.
            if hasattr(runner.alg, "policy"):
                policy_nn = runner.alg.policy
            elif hasattr(runner.alg, "actor_critic"):
                policy_nn = runner.alg.actor_critic
            elif hasattr(runner.alg, "actor"):
                policy_nn = _PolicyExportWrapper(runner.alg.actor)
            else:
                policy_nn = None

            # Extract the normalizer.
            if policy_nn is None:
                normalizer = None
            elif hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            if policy_nn is None:
                raise RuntimeError("Could not find a policy module to export.")
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported policy to: {export_model_dir}")
    except Exception as exc:
        print(f"[WARNING] Failed to export policy, continuing playback: {exc}")
    
    dt = env.unwrapped.step_dt


    # Reset environment
    obs = env.get_observations()
    timestep = 0

    # Play loop
    print(f"[INFO] Starting policy playback...")
    while simulation_app.is_running():
        start_time = time.time()

        # Run policy inference
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == 1 or timestep % 200 == 0:
                print(f"[PROGRESS] Recorded {timestep}/{args_cli.video_length} frames")
            
            # Exit after recording specified length
            if timestep == args_cli.video_length:
                break
        
        # Time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    
    # Close environment
    env.close()
    print(f"[INFO] Playback complete.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
