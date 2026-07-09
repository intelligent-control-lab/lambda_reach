# file: my_height_ball_events.py
import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.events import push_by_setting_velocity as _base_push_by_setting_velocity
from isaaclab.assets import Articulation, DeformableObject, RigidObject

def push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_gaussian: dict[str, dict[str, float]] = None,
    saturation: dict[str, tuple[float, float]] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Sample push velocity from a Gaussian distribution with configurable mean, variance, and saturation.
    velocity_gaussian: dict with keys for each axis (e.g., 'x', 'y'), each value is a dict with 'mean' and 'std'.
    saturation: dict with keys for each axis, each value is a (min, max) tuple for clamping.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Default to zero mean, unit std if not provided
    if velocity_gaussian is None:
        velocity_gaussian = {k: {'mean': 0.0, 'std': 1.0} for k in ['x', 'y']}
    if saturation is None:
        saturation = {k: (-3.0, 3.0) for k in ['x', 'y']}

    # Get asset
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # Get current root velocity for selected envs
    vel_w = asset.data.root_vel_w[env_ids]

    # All 6 axes: x, y, z, roll, pitch, yaw
    axes = ["x", "y", "z", "roll", "pitch", "yaw"]
    axis_map = {k: i for i, k in enumerate(axes)}

    # For each axis, sample Gaussian or set to 0 if not specified, and ADD to current vel_w
    for k in axes:
        if k in velocity_gaussian:
            mean = velocity_gaussian[k].get('mean', 0.0)
            std = velocity_gaussian[k].get('std', 1.0)
            min_val, max_val = saturation.get(k, (-3.0, 3.0))
            size = (vel_w.shape[0],)
            v = torch.normal(mean=mean, std=std, size=size, device=vel_w.device)
            # rejection sampling: resample out-of-bounds values
            mask = (v < min_val) | (v > max_val)
            max_attempts = 10
            attempts = 0
            while mask.any() and attempts < max_attempts:
                n_resample = mask.sum().item()
                v_new = torch.normal(mean=mean, std=std, size=(n_resample,), device=vel_w.device)
                v[mask] = v_new
                mask = (v < min_val) | (v > max_val)
                attempts += 1
            # If still out-of-bounds after max_attempts, set those to mean value
            mask = (v < min_val) | (v > max_val)
            if mask.any():
                v[mask] = mean
            vel_w[:, axis_map[k]] += v
        # else: do not modify vel_w for axes not specified

    # Set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    if not hasattr(env, "_event_mask_push"):
        env._event_mask_push = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._event_mask_push[env_ids] = True


def pin_joints_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    write_state: bool = True,
):
    """Pin selected joints to their default positions with zero velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    joint_pos = asset.data.default_joint_pos[env_ids][:, joint_ids]
    joint_vel = torch.zeros_like(joint_pos)

    asset.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, joint_ids=joint_ids, env_ids=env_ids)

    if write_state:
        asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)

# ---------------------------------------------------------------------------- #
#                                  general viz                                 #
# ---------------------------------------------------------------------------- #

def ball_viz_setup(env: ManagerBasedRLEnv, env_ids, viz, n_bins, color_range, radius):
    '''
        color_range: tuple of two RGB tuples, e.g., ((0,1,0), (1,0,0)) for green to red
    '''
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

    if (
        hasattr(env, viz)
        and hasattr(env, signal_category_attr)
        and signal_name in getattr(env, signal_category_attr)
    ):
        getattr(env, viz).visualize(
            translations=getattr(env, signal_category_attr)[signal_name]['pos'],
            marker_indices=getattr(env, signal_category_attr)[signal_name]['bin'],
        )