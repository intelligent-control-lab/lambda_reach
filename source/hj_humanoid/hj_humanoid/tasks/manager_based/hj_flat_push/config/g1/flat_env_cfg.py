# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import hj_humanoid.tasks.manager_based.hj_flat_push.mdp as mdp

from isaaclab_assets import G1_29DOF_CFG
from isaaclab.envs.mdp import push_by_setting_velocity as isaaclab_push_by_setting_velocity
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg as _G1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg
from hj_humanoid.tasks.manager_based.hj_flat_push.mdp.rewards import (
    energy as unitree_energy,
    feet_gait as unitree_feet_gait,
    foot_clearance_reward as unitree_foot_clearance_reward,
)


G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_HAND_JOINT_NAMES = [
    ".*_index_.*",
    ".*_middle_.*",
    ".*_thumb_.*",
]

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Local copy of the Unitree RL Lab velocity command cfg with curriculum limit ranges."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class UnitreeCurriculumCfg:
    """Curriculum terms used only by the UNITREE flat baseline."""

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


G1_29DOF_LOCOMOTION_ACTUATORS = {
    "N7520-14.3": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
        effort_limit_sim=88,
        velocity_limit_sim=32.0,
        stiffness={
            ".*_hip_.*": 100.0,
            "waist_yaw_joint": 200.0,
        },
        damping={
            ".*_hip_.*": 2.0,
            "waist_yaw_joint": 5.0,
        },
        armature=0.01,
    ),
    "N7520-22.5": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
        effort_limit_sim=139,
        velocity_limit_sim=20.0,
        stiffness={
            ".*_hip_roll_.*": 100.0,
            ".*_knee_.*": 150.0,
        },
        damping={
            ".*_hip_roll_.*": 2.0,
            ".*_knee_.*": 4.0,
        },
        armature=0.01,
    ),
    "N5020-16": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_.*",
            ".*_elbow_.*",
            ".*_wrist_roll.*",
            ".*_ankle_.*",
            "waist_roll_joint",
            "waist_pitch_joint",
        ],
        effort_limit_sim=25,
        velocity_limit_sim=37,
        stiffness=40.0,
        damping={
            ".*_shoulder_.*": 1.0,
            ".*_elbow_.*": 1.0,
            ".*_wrist_roll.*": 1.0,
            ".*_ankle_.*": 2.0,
            "waist_.*_joint": 5.0,
        },
        armature=0.01,
    ),
    "W4010-25": ImplicitActuatorCfg(
        joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
        effort_limit_sim=5,
        velocity_limit_sim=22,
        stiffness=40.0,
        damping=1.0,
        armature=0.01,
    ),
    "hands": ImplicitActuatorCfg(
        joint_names_expr=G1_HAND_JOINT_NAMES,
        effort_limit_sim=300,
        velocity_limit_sim=100,
        stiffness=40,
        damping=10,
        armature=0.001,
    ),
}


@configclass
class G1Flat29DofDeployPolicyObsCfg(ObsGroup):
    """Actor observations kept deployable in unitree_rl_lab: no privileged base linear velocity."""

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=G1_29DOF_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    joint_vel_rel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=G1_29DOF_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.history_length = 5
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class G1Flat29DofPrivilegedCriticObsCfg(ObsGroup):
    """Critic observations for asymmetric PPO; deployment still uses only the policy group."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=G1_29DOF_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    joint_vel_rel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=G1_29DOF_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.history_length = 5
        self.enable_corruption = False
        self.concatenate_terms = True


def configure_g1_29dof_asymmetric_io(env_cfg):
    """Switch a 29-DoF env to deployable actor observations plus privileged critic observations."""

    env_cfg.observations.policy = G1Flat29DofDeployPolicyObsCfg()
    env_cfg.observations.critic = G1Flat29DofPrivilegedCriticObsCfg()

    if getattr(env_cfg.actions, "joint_pos", None) is not None:
        env_cfg.actions.JointPositionAction = copy.deepcopy(env_cfg.actions.joint_pos)
        env_cfg.actions.joint_pos = None


def configure_g1_29dof_robot(env_cfg):
    """Apply the common 29-DoF G1 robot, action, and joint-observation layout."""

    robot_cfg = copy.deepcopy(G1_29DOF_CFG)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.activate_contact_sensors = True
    robot_cfg.spawn.articulation_props.enabled_self_collisions = True
    robot_cfg.init_state.pos = (0.0, 0.0, 0.8)
    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot_cfg.init_state.joint_pos.update(
        {
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
            "left.*thumb_2_joint": math.pi / 2.0,
            "right.*thumb_2_joint": -math.pi / 2.0,
        }
    )
    robot_cfg.actuators = copy.deepcopy(G1_29DOF_LOCOMOTION_ACTUATORS)
    env_cfg.scene.robot = robot_cfg

    env_cfg.actions.joint_pos.joint_names = G1_29DOF_JOINT_NAMES
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.actions.joint_pos.scale = {
        ".*_hip_.*_joint": 0.25,
        ".*_knee_joint": 0.25,
        ".*_ankle_.*": 0.25,
        "waist_yaw_joint": 0.25,
        "waist_roll_joint": 0.02,
        "waist_pitch_joint": 0.0,
        ".*_shoulder_.*_joint": 0.25,
        ".*_elbow_joint": 0.25,
        ".*_wrist_.*_joint": 0.0,
    }

    env_cfg.observations.policy.joint_pos.params = {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_JOINT_NAMES,
            preserve_order=True,
        )
    }
    env_cfg.observations.policy.joint_vel.params = {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_JOINT_NAMES,
            preserve_order=True,
        )
    }


# ---------------------------------------------------------------------------- #
#                                 Policy Train                                 #
# ---------------------------------------------------------------------------- #

@configclass
class G1FlatEnvCfg_POLICY(_G1FlatEnvCfg):
    """
    Env used to train the RL policy to be certified.
    """

    pass


@configclass
class G1Flat29DofEnvCfg_UNITREE(_G1FlatEnvCfg):
    """
    29-DoF G1 flat env that mirrors the unitree_rl_lab locomotion recipe while staying isolated from
    the existing HJ policy cfgs.
    """

    def __post_init__(self):
        super().__post_init__()

        configure_g1_29dof_robot(self)
        configure_g1_29dof_asymmetric_io(self)

        joint_asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_JOINT_NAMES,
            preserve_order=True,
        )
        feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")
        feet_asset_cfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*")

        # Match unitree_rl_lab's original training default joint pose. This is the reference used by
        # use_default_offset=True, joint_pos_rel, and joint_deviation_l1.
        joint_pos_defaults = copy.deepcopy(self.scene.robot.init_state.joint_pos)
        for inherited_key in (
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            ".*_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
        ):
            joint_pos_defaults.pop(inherited_key, None)
        joint_pos_defaults.update(
            {
                "left_hip_pitch_joint": -0.1,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.1,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_joint": 0.3,
                "right_ankle_pitch_joint": -0.2,
                "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.3,
                "left_shoulder_roll_joint": 0.25,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.97,
                "left_wrist_roll_joint": 0.15,
                "left_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_shoulder_pitch_joint": 0.3,
                "right_shoulder_roll_joint": -0.25,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.97,
                "right_wrist_roll_joint": -0.15,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
            }
        )
        self.scene.robot.init_state.joint_pos = joint_pos_defaults

        # Match unitree_rl_lab training action layout exactly for the 29-DoF actor.
        self.actions.JointPositionAction.joint_names = G1_29DOF_JOINT_NAMES
        self.actions.JointPositionAction.preserve_order = True
        self.actions.JointPositionAction.scale = 0.25

        # Unitree-style velocity command distribution.
        self.commands.base_velocity = UniformLevelVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=False,
            debug_vis=True,
            ranges=UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.1, 0.1),
                lin_vel_y=(-0.1, 0.1),
                ang_vel_z=(-0.1, 0.1),
            ),
            limit_ranges=UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.2, 0.2),
            ),
        )
        self.curriculum = UnitreeCurriculumCfg()

        # Match unitree_rl_lab event ranges on the existing flat scene.
        self.events.physics_material.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-1.0, 3.0),
                "operation": "add",
            },
        )
        self.events.base_com = None
        self.events.base_external_force_torque.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        }
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params = {
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        }
        self.events.push_robot = EventTerm(
            func=isaaclab_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 5.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        # Disable the existing HJ/IsaacLab flat locomotion reward recipe and replace it with the
        # unitree_rl_lab G1 locomotion reward structure.
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.termination_penalty = None
        self.rewards.dof_pos_limits = None
        self.rewards.feet_slide = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.undesired_contacts = None
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_shoulder_pitch = None
        self.rewards.joint_deviation_shoulder_roll = None
        self.rewards.joint_deviation_knee = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None

        self.rewards.track_lin_vel_xy = RewTerm(
            func=velocity_mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.track_ang_vel_z = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)
        self.rewards.base_linear_velocity = RewTerm(
            func=mdp.lin_vel_z_l2,
            weight=-2.0,
            params={"asset_cfg": joint_asset_cfg},
        )
        self.rewards.base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        self.rewards.joint_vel = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.001,
            params={"asset_cfg": joint_asset_cfg},
        )
        self.rewards.joint_acc = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": joint_asset_cfg},
        )
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-5.0,
            params={"asset_cfg": joint_asset_cfg},
        )
        self.rewards.energy = RewTerm(
            func=unitree_energy,
            weight=-2e-5,
            params={"asset_cfg": joint_asset_cfg},
        )
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
                )
            },
        )
        self.rewards.joint_deviation_waists = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
        )
        self.rewards.joint_deviation_legs = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
        )
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-10.0,
            params={"target_height": 0.78},
        )
        self.rewards.gait = RewTerm(
            func=unitree_feet_gait,
            weight=0.5,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": feet_sensor_cfg,
            },
        )
        self.rewards.feet_slide = RewTerm(
            func=velocity_mdp.feet_slide,
            weight=-0.2,
            params={"asset_cfg": feet_asset_cfg, "sensor_cfg": feet_sensor_cfg},
        )
        self.rewards.feet_clearance = RewTerm(
            func=unitree_foot_clearance_reward,
            weight=1.0,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "asset_cfg": feet_asset_cfg,
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
            },
        )

        # Replace HJ-specific termination logic with the unitree_rl_lab termination recipe.
        self.terminations.base_contact = None
        self.terminations.hand_contact = None
        self.terminations.upper_body_height = None
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.8},
        )


@configclass
class G1Flat29DofEnvCfg_UNITREE_PLAY(G1Flat29DofEnvCfg_UNITREE):
    """Play-time UNITREE cfg that mirrors unitree_rl_lab's separate play env behavior."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.rel_standing_envs = 0.0


# ---------------------------------------------------------------------------- #
#                                   HJ Learn                                   #
# ---------------------------------------------------------------------------- #

@configclass
class G1FlatEnvRecorderManagerCfg(RecorderManagerBaseCfg):

    record_safety_signal_balance = mdp.SafetySigBalanceRecorderCfg(
        class_type=mdp.SafetySigBalanceRecorder,
        asset=SceneEntityCfg("robot"),
        bin_edges=(-0.5, 0.0, 0.5),
        key_prefix="safety_signal_balance",
        ball_offset=(0.0, 0.0, 0.8),
        phi_max=np.pi / 6.0,
        contact_sensor_cfg=SceneEntityCfg("contact_forces", body_names="torso_link"),
        contact_threshold=1.0,
        stance_clearance_min=0.35,
    )

    record_stability_signal_vel_track = mdp.StabilitySigVelTrackRecorderCfg(
        class_type=mdp.StabilitySigVelTrackRecorder,
        asset=SceneEntityCfg("robot"),
        bin_edges=(0.25, 0.5, 0.75),
        key_prefix="stability_signal_vel_track",
        ball_offset_tracking=(-0.2, 0.0, 0.8),
        ball_offset_stability=(-0.4, 0.0, 0.8),
        stable_steps=10,
    )

    record_push_events = mdp.EventPushRecorderCfg(
        class_type=mdp.EventPushRecorder,
        key_prefix="event_push",
    )

    record_terminal_states = mdp.TerminalStateRecorderCfg(
        class_type=mdp.TerminalStateRecorder,
        key_prefix="terminal_state",
    )

    record_observations = mdp.ObservationRecorderCfg(
        class_type=mdp.ObservationRecorder,
        key_prefix="policy_obs",
    )


@configclass
class G1FlatEnvCfg_SAFETY_ROLLOUT(G1FlatEnvCfg_POLICY):
    """
    Env used to generate policy rollouts for HJ training.
    """

    recorders: G1FlatEnvRecorderManagerCfg = G1FlatEnvRecorderManagerCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # --------------------------- add safety signal viz -------------------------- #
        viz_safety_signal_balance = "safety_signal_balance"
        self.events.safety_signal_balance_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_signal_balance,
                n_bins=4,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # green to red
                radius=0.07,
            ),
            mode="startup",
        )
        self.events.safety_signal_balance_viz_update = EventTerm(
            func=mdp.ball_viz_update,
            params=dict(
                viz=viz_safety_signal_balance,
                signal_category_attr="safety_signal",
                signal_name="balance",
            ),
            mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        )

        # ---- recreate (override) the push event with Gaussian velocity sampling ---- #
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 4.0),
            params={
                "velocity_gaussian": {
                    "x": {"mean": 0.0, "std": 1.0},
                    "y": {"mean": 0.0, "std": 1.0},
                },
                "saturation": {
                    "x": (-3.0, 3.0),
                    "y": (-3.0, 3.0),
                },
            },
        )

        # ----------------------- constant command per episode ----------------------- #
        self.commands.base_velocity.resampling_time_range = (100000000, 100000000)


@configclass
class G1Flat29DofEnvCfg_UNITREE_SAFETY_ROLLOUT(G1Flat29DofEnvCfg_UNITREE):
    """
    Safety rollout env for a Unitree-style 29-DoF policy checkpoint.
    """

    recorders: G1FlatEnvRecorderManagerCfg = G1FlatEnvRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()

        # --------------------------- add safety signal viz -------------------------- #
        viz_safety_signal_balance = "safety_signal_balance"
        self.events.safety_signal_balance_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_signal_balance,
                n_bins=4,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # green to red
                radius=0.07,
            ),
            mode="startup",
        )
        self.events.safety_signal_balance_viz_update = EventTerm(
            func=mdp.ball_viz_update,
            params=dict(
                viz=viz_safety_signal_balance,
                signal_category_attr="safety_signal",
                signal_name="balance",
            ),
            mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        )

        # Use the local push event so EventPushRecorder can mark segment boundaries.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 4.0),
            params={
                "velocity_gaussian": {
                    "x": {"mean": 0.0, "std": 1.0},
                    "y": {"mean": 0.0, "std": 1.0},
                },
                "saturation": {
                    "x": (-3.0, 3.0),
                    "y": (-3.0, 3.0),
                },
            },
        )

        # Match the final Unitree command curriculum range used by the deployed controller.
        self.commands.base_velocity.ranges = copy.deepcopy(self.commands.base_velocity.limit_ranges)

        # ----------------------- constant command per episode ----------------------- #
        self.commands.base_velocity.resampling_time_range = (100000000, 100000000)

# ---------------------------------------------------------------------------- #
#                                    HJ Play                                   #
# ---------------------------------------------------------------------------- #
@configclass
class G1FlatEnvCfg_SAFETY_INFERENCE(G1FlatEnvCfg_SAFETY_ROLLOUT):
    """
    Env used to verify safety value by comparing safety sign and max l value in finite look-ahead (normally)
    """

    # Path to safety value directory (set via command-line argument)
    safety_value_path: str = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Add safety value predictor to the existing recorders
        if not hasattr(self.recorders, "record_safety_value"):
            self.recorders.record_safety_value = mdp.SafetyValueInferenceCfg(
                class_type=mdp.SafetyValueInference,
                asset=SceneEntityCfg("robot"),
                bin_edges=(-0.5, 0.0),
                key_prefix="safety_value",
                ball_offset=(0.0, 0.0, 1.2),
            )

        # Add safety value visualization events
        viz_safety_value = "safety_value_prediction"
        self.events.safety_value_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_value,
                n_bins=3,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # green to red
                radius=0.07,
            ),
            mode="startup",
        )
        self.events.safety_value_viz_update = EventTerm(
            func=mdp.ball_viz_update,
            params=dict(
                viz=viz_safety_value,
                signal_category_attr="safety_value",
                signal_name="prediction",
            ),
            mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        )


@configclass
class G1Flat29DofEnvCfg_UNITREE_SAFETY_INFERENCE(G1Flat29DofEnvCfg_UNITREE_SAFETY_ROLLOUT):
    """
    Safety inference env for a Unitree-style 29-DoF policy checkpoint.
    """

    # Path to safety value directory (set via command-line argument)
    safety_value_path: str = None

    def __post_init__(self):
        super().__post_init__()

        # Add safety value predictor to the existing recorders
        if not hasattr(self.recorders, "record_safety_value"):
            self.recorders.record_safety_value = mdp.SafetyValueInferenceCfg(
                class_type=mdp.SafetyValueInference,
                asset=SceneEntityCfg("robot"),
                bin_edges=(-0.5, 0.0),
                key_prefix="safety_value",
                ball_offset=(0.0, 0.0, 1.2),
            )

        # Add safety value visualization events
        viz_safety_value = "safety_value_prediction"
        self.events.safety_value_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_value,
                n_bins=3,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # green to red
                radius=0.07,
            ),
            mode="startup",
        )
        self.events.safety_value_viz_update = EventTerm(
            func=mdp.ball_viz_update,
            params=dict(
                viz=viz_safety_value,
                signal_category_attr="safety_value",
                signal_name="prediction",
            ),
            mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        )
