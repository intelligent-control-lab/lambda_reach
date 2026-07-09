# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for collision avoidance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_unless_ball_enabled(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height_for_ball_hit: float = 0.4,
) -> torch.Tensor:
    """
    Terminate when contact force exceeds threshold AND robot is fallen (low height).
    
    Ball hits on torso while robot is standing should NEVER terminate the episode,
    regardless of curriculum state. This allows consistent behavior during both
    warmup (ball disabled) and collision avoidance learning (ball enabled).
    
    Termination logic:
        - Contact + robot height < min_height_for_ball_hit: terminate (robot fallen)
        - Contact + robot height >= min_height_for_ball_hit: don't terminate (ball hit while standing)
    
    Args:
        env: The environment instance.
        threshold: Force magnitude threshold for termination.
        sensor_cfg: Scene entity config for contact sensor.
        robot_cfg: Scene entity config for the robot.
        min_height_for_ball_hit: Minimum robot base height to consider contact as ball hit.
            Below this height, robot is considered fallen and should terminate.
        
    Returns:
        Boolean tensor (num_envs,) - True if should terminate.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # Check if any contact force exceeds the threshold
    # Shape: (num_envs, history_length, num_bodies, 3)
    force_magnitudes = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    max_force = torch.max(force_magnitudes, dim=1)[0]  # max over history
    max_force = torch.max(max_force, dim=1)[0]  # max over bodies
    
    has_contact = max_force > threshold
    
    # Always check robot height - ball hits while standing should never terminate
    robot: Articulation = env.scene[robot_cfg.name]
    robot_height = robot.data.root_pos_w[:, 2]  # (num_envs,)
    
    # Robot is fallen if height is below threshold
    is_fallen = robot_height < min_height_for_ball_hit
    
    # Only terminate if contact AND robot is fallen (not just ball hit while standing)
    return has_contact & is_fallen
