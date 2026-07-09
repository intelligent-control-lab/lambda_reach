# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory where recorder datasets and videos should be stored during play.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help="Dataset filename (without extension) to use for recorded episodes during play.",
)
parser.add_argument(
    "--hj_value_path",
    type=str,
    default=None,
    help="Path to HJ value directory (e.g., logs/hj/hj_g1_flat_push/hj_value_v1) for visualization.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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

import copy
import importlib.metadata as metadata
import os
import time

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
from isaaclab.utils.dict import print_dict

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import hj_humanoid.tasks  # noqa: F401


class _PolicyExportWrapper(torch.nn.Module):
    """Adapts newer rsl-rl actor models to Isaac Lab's legacy policy exporter API."""

    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        self.actor = actor
        self.is_recurrent = getattr(actor, "is_recurrent", False)


def _has_base_velocity_hydra_override() -> bool:
    """Return True when the user explicitly set base velocity through Hydra CLI args."""

    for arg in hydra_args:
        normalized_arg = arg.lstrip("+")
        if normalized_arg.startswith("env.commands.base_velocity."):
            return True
    return False


def _apply_play_velocity_ranges(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, task_name: str):
    """Use play-specific velocity ranges when the task registers a play env cfg."""

    if _has_base_velocity_hydra_override():
        print("[INFO] Keeping user-provided env.commands.base_velocity Hydra overrides.")
        return

    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
        return

    try:
        play_env_cfg = load_cfg_from_registry(task_name, "play_env_cfg_entry_point")
    except Exception:
        return

    if not hasattr(play_env_cfg, "commands") or not hasattr(play_env_cfg.commands, "base_velocity"):
        return

    command_cfg = env_cfg.commands.base_velocity
    play_command_cfg = play_env_cfg.commands.base_velocity
    command_cfg.ranges = copy.deepcopy(play_command_cfg.ranges)
    command_cfg.rel_standing_envs = play_command_cfg.rel_standing_envs
    if hasattr(command_cfg, "limit_ranges") and hasattr(play_command_cfg, "limit_ranges"):
        command_cfg.limit_ranges = copy.deepcopy(play_command_cfg.limit_ranges)

    ranges = command_cfg.ranges
    print(
        "[INFO] Using play velocity ranges: "
        f"lin_vel_x={ranges.lin_vel_x}, lin_vel_y={ranges.lin_vel_y}, ang_vel_z={ranges.ang_vel_z}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    _apply_play_velocity_ranges(env_cfg, task_name)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    default_log_dir = os.path.dirname(resume_path)
    if args_cli.output_dir:
        log_dir = os.path.abspath(args_cli.output_dir)
    else:
        log_dir = default_log_dir
    
    # When using HJ value prediction, organize outputs by seed
    if args_cli.hj_value_path is not None:
        seed_suffix = f"seed_{agent_cfg.seed}"
        log_dir = os.path.join(log_dir, seed_suffix)
        print(f"[INFO] HJ value prediction enabled - organizing outputs by seed: {seed_suffix}")
    
    os.makedirs(log_dir, exist_ok=True)

    if getattr(env_cfg, "recorders", None) is not None:
        if args_cli.output_dir or args_cli.hj_value_path is not None:
            env_cfg.recorders.dataset_export_dir_path = log_dir
        if args_cli.dataset_name:
            env_cfg.recorders.dataset_filename = args_cli.dataset_name

    # pass HJ value path to environment if provided
    if args_cli.hj_value_path is not None:
        env_cfg.hj_value_path = os.path.abspath(args_cli.hj_value_path)
        print(f"[INFO] HJ value path set to: {env_cfg.hj_value_path}")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        if "SAFETY-ROLLOUT" in args_cli.task:
            video_subdir = "rollout"
        elif "SAFETY-INFERENCE" in args_cli.task:
            video_subdir = "inference"
        else:
            video_subdir = "play"
        
        video_folder = os.path.join(log_dir, "videos", video_subdir)
        
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos during training. Saving to: {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        if hasattr(runner, "export_policy_to_jit") and hasattr(runner, "export_policy_to_onnx"):
            runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
        else:
            # Resolve the underlying module for older Isaac Lab / rsl-rl combinations.
            if hasattr(runner.alg, "policy"):
                policy_nn = runner.alg.policy
            elif hasattr(runner.alg, "actor_critic"):
                policy_nn = runner.alg.actor_critic
            elif hasattr(runner.alg, "actor"):
                policy_nn = _PolicyExportWrapper(runner.alg.actor)
            else:
                policy_nn = None

            if policy_nn is None:
                raise RuntimeError("Could not find a policy module to export.")

            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported policy to: {export_model_dir}")
    except Exception as exc:
        print(f"[WARNING] Failed to export policy, continuing playback: {exc}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == 1 or timestep % 200 == 0:
                print(f"[PROGRESS] Recorded {timestep}/{args_cli.video_length} frames")
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    
    # Generate plots if HJ value prediction was used
    if args_cli.hj_value_path is not None and args_cli.video:
        print("\n" + "="*80)
        print("Generating HJ value plots...")
        print("="*80)
        
        try:
            # Import plot function
            import sys
            dpe_scripts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "dpe", "scripts")
            if dpe_scripts_path not in sys.path:
                sys.path.insert(0, dpe_scripts_path)
            
            from plot_play_data import plot_all_episodes
            
            # Plot all episodes to seed folder/plots
            plot_output_dir = os.path.join(log_dir, "plots")
            
            # The play_data.hdf5 is saved in log_dir by the recorder
            # We need to pass log_dir as the data_dir for plotting
            print(f"[INFO] Plotting all episodes to: {plot_output_dir}")
            plot_all_episodes(data_dir=log_dir, output_dir=plot_output_dir)
            
            print(f"[INFO] Successfully generated plots in: {plot_output_dir}")
        except Exception as e:
            print(f"[WARNING] Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
