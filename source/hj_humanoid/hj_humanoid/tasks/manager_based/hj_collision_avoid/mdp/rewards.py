# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for collision avoidance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_collision_penalty(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    threshold: float = 1.0,
    min_height_for_ground_filter: float | None = None,
) -> torch.Tensor:
    """
    Penalize collision between ball and robot bodies.
    
    Uses contact sensor on robot bodies to detect collision forces above threshold.
    Returns 1.0 for envs with collision, 0.0 otherwise (to be weighted negatively).
    
    Args:
        env: The environment instance.
        contact_sensor_cfg: Scene entity config for contact sensor on robot bodies.
        threshold: Force magnitude threshold for collision detection.
        
    Returns:
        Collision penalty (num_envs,) - 1.0 if collision, 0.0 otherwise.
    """
    # Return zero penalty if ball is not yet enabled by curriculum
    if not getattr(env, "_ball_enabled", False):
        return torch.zeros(env.num_envs, device=env.device)
    
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # Get net contact forces on robot bodies
    # Shape: (num_envs, T, num_bodies, 3) where T is history length
    net_forces = contact_sensor.data.net_forces_w_history
    
    # Compute force magnitude and check if any exceeds threshold
    forces_magnitude = net_forces.norm(dim=-1)  # (num_envs, T, num_bodies)
    max_force = forces_magnitude.max(dim=1)[0].max(dim=1)[0]  # (num_envs,)
    
    has_collision = max_force > threshold

    if min_height_for_ground_filter is not None:
        ball: RigidObject = env.scene[ball_cfg.name]
        is_above_ground = ball.data.root_pos_w[:, 2] > min_height_for_ground_filter
        has_collision = has_collision & is_above_ground
    
    return has_collision.float()


def ball_proximity_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
) -> torch.Tensor:
    """
    Penalize robot for being close to the ball.
    
    Returns exponential penalty based on distance: exp(-distance / std).
    Penalty is highest (1.0) when ball is at robot position, decreases with distance.
    Returns 0 if ball is not enabled by curriculum.
    
    Args:
        env: The environment instance.
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        std: Standard deviation controlling penalty falloff rate.
        
    Returns:
        Proximity penalty (num_envs,) in range [0, 1].
    """
    # Return zero penalty if ball is not yet enabled by curriculum
    if not getattr(env, "_ball_enabled", False):
        return torch.zeros(env.num_envs, device=env.device)
    
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get positions
    ball_pos = ball.data.root_pos_w  # (num_envs, 3)
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    
    # Compute distance
    distance = (ball_pos - robot_pos).norm(dim=1)  # (num_envs,)
    
    # Exponential penalty
    penalty = torch.exp(-distance / std)
    
    return penalty


def ball_approaching_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_speed: float = 6.0,
    danger_distance: float = 2.0,
) -> torch.Tensor:
    """
    Penalize when ball is approaching the robot (negative radial velocity).
    
    This encourages the robot to move away from incoming balls before they get close.
    Penalty is normalized to [0, 1] range based on:
    - approaching_speed / max_speed (how fast ball is coming)
    - danger_distance / distance (how close the ball is)
    
    Returns 0 if ball is not enabled by curriculum.
    
    Args:
        env: The environment instance.
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        max_speed: Maximum expected ball speed for normalization.
        danger_distance: Distance at which penalty factor = 1.
        
    Returns:
        Approaching penalty (num_envs,) in range [0, 1].
    """
    # Return zero penalty if ball is not yet enabled by curriculum
    if not getattr(env, "_ball_enabled", False):
        return torch.zeros(env.num_envs, device=env.device)
    
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get positions and velocities
    ball_pos = ball.data.root_pos_w  # (num_envs, 3)
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    ball_vel = ball.data.root_lin_vel_w  # (num_envs, 3)
    robot_vel = robot.data.root_lin_vel_w  # (num_envs, 3)
    
    # Relative position and velocity (ball relative to robot)
    rel_pos = ball_pos - robot_pos  # (num_envs, 3)
    rel_vel = ball_vel - robot_vel  # (num_envs, 3)
    
    # Distance
    distance = rel_pos.norm(dim=1).clamp(min=0.1)  # (num_envs,)
    
    # Direction from robot to ball (unit vector)
    direction = rel_pos / distance.unsqueeze(1)  # (num_envs, 3)
    
    # Radial velocity (positive = moving away, negative = approaching)
    radial_vel = (rel_vel * direction).sum(dim=1)  # (num_envs,)
    
    # Penalty for approaching (negative radial velocity)
    # Normalize speed to [0, 1]
    approaching_speed = torch.clamp(-radial_vel, min=0.0)  # (num_envs,)
    speed_factor = torch.clamp(approaching_speed / max_speed, max=1.0)  # [0, 1]
    
    # Distance factor: 1.0 at danger_distance, higher when closer, capped at 1.0 when far
    distance_factor = torch.clamp(danger_distance / distance, max=1.0)  # [0, 1]
    
    # Combined penalty in [0, 1]
    penalty = speed_factor * distance_factor
    
    return penalty


def ball_distance_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward robot for maintaining distance from the ball.
    
    Returns the Euclidean distance between ball and robot root.
    Meant to be weighted positively to encourage keeping distance.
    
    Args:
        env: The environment instance.
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        
    Returns:
        Distance from ball (num_envs,).
    """
    # Return zero if ball is not enabled by curriculum
    if not getattr(env, "_ball_enabled", False):
        return torch.zeros(env.num_envs, device=env.device)
    
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    ball_pos = ball.data.root_pos_w
    robot_pos = robot.data.root_pos_w
    
    distance = (ball_pos - robot_pos).norm(dim=1)
    
    return distance
