# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for collision avoidance task."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import omni.usd
from pxr import Sdf, Gf

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate, quat_from_euler_xyz
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------- #
#                              Visualization Functions                          #
# ---------------------------------------------------------------------------- #

def ball_viz_setup(env: ManagerBasedRLEnv, env_ids, viz, n_bins, color_range, radius):
    """
    Setup ball visualization markers for safety signal display.
    
    Creates colored spheres for each environment that can be updated to show
    the current safety signal value (green = safe, red = unsafe).
    
    Args:
        env: The environment instance.
        env_ids: Environment indices (not used, but required by event signature).
        viz: Name of the visualization attribute to create on env.
        n_bins: Number of color bins.
        color_range: Tuple of two RGB tuples for color gradient, e.g., ((0,1,0), (1,0,0)).
        radius: Radius of the visualization spheres.
    """
    n_bins = int(n_bins)
    if hasattr(env, viz):
        return

    colors = torch.stack(
        [torch.linspace(color_range[0][0], color_range[1][0], int(n_bins), device=env.device),
         torch.linspace(color_range[0][1], color_range[1][1], int(n_bins), device=env.device),
         torch.linspace(color_range[0][2], color_range[1][2], int(n_bins), device=env.device)], dim=1
    )

    marker_dict = {
        f"bin{i:02d}": sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(colors[i].tolist())),
        )
        for i in range(int(n_bins))
    }

    viz_cfg = VisualizationMarkersCfg(
        prim_path=f"/Visuals/{viz}_balls",
        markers=marker_dict,
    )

    setattr(env, viz, VisualizationMarkers(viz_cfg))


@torch.no_grad()
def ball_viz_update(env, env_ids, viz, signal_category_attr, signal_name):
    """
    Update ball visualization positions and colors based on safety signal.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices (not used, but required by event signature).
        viz: Name of the visualization attribute on env.
        signal_category_attr: Name of the env attribute containing signal dict (e.g., "safety_signal").
        signal_name: Key in the signal dict for this specific signal (e.g., "collision").
    """
    if (
        hasattr(env, viz)
        and hasattr(env, signal_category_attr)
        and signal_name in getattr(env, signal_category_attr)
    ):
        getattr(env, viz).visualize(
            translations=getattr(env, signal_category_attr)[signal_name]['pos'],
            marker_indices=getattr(env, signal_category_attr)[signal_name]['bin'],
        )


def update_ball_color_on_contact(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("ball_contact"),
    contact_threshold: float = 0.1,
    min_height_for_ground_filter: float = 0.15,
):
    """
    Update ball visual color based on contact detection with robot.
    
    Ball turns red when contact is detected (ball above ground), white otherwise.
    This provides visual feedback that the contact sensor is detecting robot hits.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices (not used, but required by event signature).
        ball_cfg: Scene entity config for the ball.
        contact_sensor_cfg: Scene entity config for contact sensor on ball.
        contact_threshold: Force threshold for contact detection.
        min_height_for_ground_filter: Minimum ball height (m) to filter out ground contacts.
            Must be higher than ball radius (0.1m) to properly filter ground hits.
    """
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    # Get net contact forces on ball
    net_forces = contact_sensor.data.net_forces_w_history  # (num_envs, T, num_bodies, 3)
    forces_magnitude = net_forces.norm(dim=-1)  # (num_envs, T, num_bodies)
    max_force = forces_magnitude.max(dim=1)[0].max(dim=1)[0]  # (num_envs,)
    
    # Detect which envs have contact
    has_contact = max_force > contact_threshold
    
    # Get ball height - filter out ground contacts
    ball_height = ball.data.root_pos_w[:, 2]  # (num_envs,)
    is_above_ground = ball_height > min_height_for_ground_filter
    
    # Robot contact = contact AND ball is above ground
    has_robot_contact = has_contact & is_above_ground
    
    # Initialize tracking if needed
    if not hasattr(env, "_ball_contact_state"):
        env._ball_contact_state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Find envs where contact state changed
    contact_changed = has_robot_contact != env._ball_contact_state
    
    if contact_changed.any():
        # Get the USD stage
        stage = omni.usd.get_context().get_stage()
        
        # Find ball prims for envs with changed state
        changed_indices = torch.where(contact_changed)[0]
        
        with Sdf.ChangeBlock():
            for idx in changed_indices:
                idx_int = int(idx.item())
                # Construct the prim path for this env's ball material
                ball_prim_path = f"/World/envs/env_{idx_int}/ObstacleBall/geometry/material/Shader"
                
                # Get the shader prim
                shader_prim = stage.GetPrimAtPath(ball_prim_path)
                if shader_prim.IsValid():
                    # Set color based on contact state
                    diffuse_attr = shader_prim.GetAttribute("inputs:diffuseColor")
                    if diffuse_attr:
                        if has_robot_contact[idx_int]:
                            # Red on robot contact
                            diffuse_attr.Set(Gf.Vec3f(1.0, 0.0, 0.0))
                        else:
                            # White when no contact (default)
                            diffuse_attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        
        # Update tracking
        env._ball_contact_state = has_robot_contact.clone()


def spawn_ball_towards_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spawn_distance_range: tuple[float, float] = (2.0, 4.0),
    spawn_height_range: tuple[float, float] = (0.5, 1.5),
    ball_speed_range: tuple[float, float] = (3.0, 8.0),
):
    """
    Spawn a ball at a random position around the robot with velocity directed towards the robot.
    
    Always spawns ball for the given env_ids at constant intervals.
    Ball is always physically present - enable/disable is handled via observations/rewards.
    
    Sets env._event_mask_ball_spawn to mark this as a segmenting event for data processing.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to spawn balls for.
        ball_cfg: Scene entity config for the ball rigid object.
        robot_cfg: Scene entity config for the robot articulation.
        spawn_distance_range: (min, max) spawn distance from robot in meters.
        spawn_height_range: (min, max) spawn height in meters.
        ball_speed_range: (min, max) initial ball speed towards robot in m/s.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    
    # Initialize spawn event mask if not present
    if not hasattr(env, "_event_mask_ball_spawn"):
        env._event_mask_ball_spawn = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Mark these envs as having a ball spawn event (for data segmentation)
    # Always mark since ball always spawns normally regardless of curriculum
    env._event_mask_ball_spawn[env_ids] = True
    
    # Get assets
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get robot root position (target point for ball to fly towards)
    robot_pos = robot.data.root_pos_w[env_ids]  # (N, 3)
    
    # Sample random spawn angle around robot (in XY plane)
    angles = torch.rand(len(env_ids), device=env.device) * 2 * math.pi  # [0, 2*pi)
    
    # Sample spawn distance
    dist_min, dist_max = spawn_distance_range
    distances = torch.rand(len(env_ids), device=env.device) * (dist_max - dist_min) + dist_min
    
    # Sample spawn height
    height_min, height_max = spawn_height_range
    heights = torch.rand(len(env_ids), device=env.device) * (height_max - height_min) + height_min
    
    # Compute spawn position in world frame
    spawn_offset_x = distances * torch.cos(angles)
    spawn_offset_y = distances * torch.sin(angles)
    
    ball_pos = torch.zeros(len(env_ids), 3, device=env.device)
    ball_pos[:, 0] = robot_pos[:, 0] + spawn_offset_x
    ball_pos[:, 1] = robot_pos[:, 1] + spawn_offset_y
    ball_pos[:, 2] = heights
    
    # Set ball orientation (identity quaternion - w, x, y, z format)
    ball_quat = torch.zeros(len(env_ids), 4, device=env.device)
    ball_quat[:, 0] = 1.0  # w = 1, x = y = z = 0
    
    # Compute velocity to hit target with gravity compensation
    # Target is robot torso at ~0.8m height
    target_pos = robot_pos.clone()
    target_pos[:, 2] = 0.8  # aim at approximate torso height
    
    # Horizontal direction (XY plane)
    delta_xy = target_pos[:, :2] - ball_pos[:, :2]  # (N, 2)
    horizontal_dist = delta_xy.norm(dim=1)  # (N,)
    
    # Sample ball speed (horizontal component)
    speed_min, speed_max = ball_speed_range
    speeds = torch.rand(len(env_ids), device=env.device) * (speed_max - speed_min) + speed_min
    
    # Time to reach target horizontally
    time_to_target = horizontal_dist / speeds  # (N,)
    
    # Compute required vertical velocity to hit target height accounting for gravity
    # Using: y = y0 + vy*t - 0.5*g*t^2
    # Solving for vy: vy = (y - y0 + 0.5*g*t^2) / t
    g = 9.81
    delta_z = target_pos[:, 2] - ball_pos[:, 2]  # (N,)
    vel_z = (delta_z + 0.5 * g * time_to_target ** 2) / time_to_target  # (N,)
    
    # Compute horizontal velocity direction
    dir_xy = delta_xy / horizontal_dist.unsqueeze(1).clamp(min=1e-6)  # (N, 2) normalized
    
    # Combine into 3D velocity
    ball_vel = torch.zeros(len(env_ids), 3, device=env.device)
    ball_vel[:, 0] = dir_xy[:, 0] * speeds
    ball_vel[:, 1] = dir_xy[:, 1] * speeds
    ball_vel[:, 2] = vel_z
    
    # Set angular velocity to zero
    ball_ang_vel = torch.zeros(len(env_ids), 3, device=env.device)
    
    # Combine pose (position + quaternion)
    ball_pose = torch.cat([ball_pos, ball_quat], dim=1)  # (N, 7)
    
    # Combine velocity (linear + angular)
    ball_velocity = torch.cat([ball_vel, ball_ang_vel], dim=1)  # (N, 6)
    
    # Write to simulation - ball always spawns normally regardless of curriculum
    # Curriculum only controls rewards, not ball physics
    ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_velocity, env_ids=env_ids)


def reset_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spawn_distance_range: tuple[float, float] = (2.0, 4.0),
    spawn_height_range: tuple[float, float] = (0.5, 1.5),
    ball_speed_range: tuple[float, float] = (3.0, 8.0),
):
    """
    Reset ball position on environment reset. Always spawns the ball for given env_ids.
    
    Ball is always physically present - enable/disable is handled via observations/rewards.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    
    # Get assets
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get robot root position
    robot_pos = robot.data.root_pos_w[env_ids]  # (N, 3)
    
    # Sample random spawn angle around robot
    angles = torch.rand(len(env_ids), device=env.device) * 2 * math.pi
    
    # Sample spawn distance
    dist_min, dist_max = spawn_distance_range
    distances = torch.rand(len(env_ids), device=env.device) * (dist_max - dist_min) + dist_min
    
    # Sample spawn height
    height_min, height_max = spawn_height_range
    heights = torch.rand(len(env_ids), device=env.device) * (height_max - height_min) + height_min
    
    # Compute spawn position
    ball_pos = torch.zeros(len(env_ids), 3, device=env.device)
    ball_pos[:, 0] = robot_pos[:, 0] + distances * torch.cos(angles)
    ball_pos[:, 1] = robot_pos[:, 1] + distances * torch.sin(angles)
    ball_pos[:, 2] = heights
    
    # Set ball orientation (identity quaternion)
    ball_quat = torch.zeros(len(env_ids), 4, device=env.device)
    ball_quat[:, 0] = 1.0
    
    # Compute velocity to hit target with gravity compensation
    # Target is robot torso at ~0.8m height
    target_pos = robot_pos.clone()
    target_pos[:, 2] = 0.8  # aim at approximate torso height
    
    # Horizontal direction (XY plane)
    delta_xy = target_pos[:, :2] - ball_pos[:, :2]  # (N, 2)
    horizontal_dist = delta_xy.norm(dim=1)  # (N,)
    
    # Sample ball speed (horizontal component)
    speed_min, speed_max = ball_speed_range
    speeds = torch.rand(len(env_ids), device=env.device) * (speed_max - speed_min) + speed_min
    
    # Time to reach target horizontally
    time_to_target = horizontal_dist / speeds  # (N,)
    
    # Compute required vertical velocity to hit target height accounting for gravity
    # Using: y = y0 + vy*t - 0.5*g*t^2
    # Solving for vy: vy = (y - y0 + 0.5*g*t^2) / t
    g = 9.81
    delta_z = target_pos[:, 2] - ball_pos[:, 2]  # (N,)
    vel_z = (delta_z + 0.5 * g * time_to_target ** 2) / time_to_target  # (N,)
    
    # Compute horizontal velocity direction
    dir_xy = delta_xy / horizontal_dist.unsqueeze(1).clamp(min=1e-6)  # (N, 2) normalized
    
    # Combine into 3D velocity
    ball_vel = torch.zeros(len(env_ids), 3, device=env.device)
    ball_vel[:, 0] = dir_xy[:, 0] * speeds
    ball_vel[:, 1] = dir_xy[:, 1] * speeds
    ball_vel[:, 2] = vel_z
    
    ball_ang_vel = torch.zeros(len(env_ids), 3, device=env.device)
    
    # Write to sim - ball always spawns normally regardless of curriculum
    # Curriculum only controls rewards, not ball physics
    ball_pose = torch.cat([ball_pos, ball_quat], dim=1)
    ball_velocity = torch.cat([ball_vel, ball_ang_vel], dim=1)
    
    ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_velocity, env_ids=env_ids)
