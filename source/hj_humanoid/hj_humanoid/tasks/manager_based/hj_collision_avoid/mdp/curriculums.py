# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for collision avoidance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def curriculum_enable_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    enable_after_iters: int = 500,
    num_steps_per_env: int = 24,  # RSL-RL default: steps collected per env per iteration
) -> None:
    """
    Curriculum function to enable/disable ball observations, rewards, and effective collision.
    
    Before enable_after_iters: 
        - observations/rewards return zeros (ball ignored by policy)
        - ball is teleported far away each step so it can't reach the robot
    After enable_after_iters: 
        - observations/rewards are active (ball affects training)
        - ball spawns normally and can collide with robot
    
    Note: We don't modify collision properties at runtime as this causes PhysX GPU errors.
    Instead, we teleport the ball far away when "disabled" to prevent any interaction.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices (not used, curriculum applies globally).
        ball_cfg: Scene entity config for the ball rigid object.
        enable_after_iters: Number of learning iterations before enabling ball signals.
        num_steps_per_env: Number of env steps per learning iteration (from RSL-RL config).
    """
    # Compute current learning iteration from step counter
    current_iter = env.common_step_counter // num_steps_per_env
    
    should_enable = current_iter >= enable_after_iters
    
    # Initialize ball enabled state if not present
    if not hasattr(env, "_ball_enabled"):
        env._ball_enabled = should_enable
        if should_enable:
            print(f"[Curriculum] Ball enabled from start (enable_after_iters={enable_after_iters})")
        else:
            print(f"[Curriculum] Ball disabled initially, will enable after {enable_after_iters} iterations")
        return
    
    # If state changed from disabled to enabled, log it
    if should_enable and not env._ball_enabled:
        print(f"[Curriculum] Enabling ball observations/rewards/collision at learning iteration {current_iter}")
        env._ball_enabled = True
