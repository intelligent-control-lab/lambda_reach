# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for collision avoidance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Compute ball position relative to robot root frame.
    
    Always returns true ball position regardless of curriculum state.
    This allows the locomotion policy to see ball observations during warmup,
    so when collision avoidance rewards are enabled, the observations don't change.
    
    Returns:
        Ball position in robot frame (num_envs, 3).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get world positions
    ball_pos_w = ball.data.root_pos_w  # (num_envs, 3)
    robot_pos_w = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4) - w, x, y, z format
    
    # Compute relative position in world frame
    relative_pos_w = ball_pos_w - robot_pos_w  # (num_envs, 3)
    
    # Transform to robot frame
    relative_pos_robot = quat_apply_inverse(robot_quat_w, relative_pos_w)
    
    return relative_pos_robot


def ball_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Compute ball velocity relative to robot root frame.
    
    Always returns true ball velocity regardless of curriculum state.
    This allows the locomotion policy to see ball observations during warmup,
    so when collision avoidance rewards are enabled, the observations don't change.
    
    Returns:
        Ball velocity in robot frame (num_envs, 3).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get world velocities
    ball_vel_w = ball.data.root_lin_vel_w  # (num_envs, 3)
    robot_vel_w = robot.data.root_lin_vel_w  # (num_envs, 3)
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4)
    
    # Compute relative velocity in world frame
    relative_vel_w = ball_vel_w - robot_vel_w  # (num_envs, 3)
    
    # Transform to robot frame
    relative_vel_robot = quat_apply_inverse(robot_quat_w, relative_vel_w)
    
    return relative_vel_robot
