# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for G1 collision avoidance task on flat terrain."""

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import hj_humanoid.tasks.manager_based.hj_collision_avoid.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg as _G1FlatEnvCfg,
)
from hj_humanoid.tasks.manager_based.hj_flat_push.config.g1.flat_env_cfg import (
    G1Flat29DofEnvCfg_UNITREE as _G1Flat29DofEnvCfg_UNITREE,
)


def _add_ball_observation_terms(observation_group):
    """Add obstacle ball state to an observation group."""

    observation_group.ball_position = ObsTerm(
        func=mdp.ball_position_in_robot_frame,
        params={
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    observation_group.ball_velocity = ObsTerm(
        func=mdp.ball_velocity_in_robot_frame,
        params={
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


def _configure_collision_avoid_policy_terms(
    env_cfg,
    *,
    enable_ball_after_iters: int = 500,
    ball_collision_weight: float = -1.0,
    ball_collision_min_height_for_ground_filter: float | None = None,
    ball_approaching_weight: float | None = None,
):
    """Append ball-avoidance assets, observations, rewards, events, and termination."""

    # -------------------- Add ball obstacle to scene -------------------- #
    # Lightweight ball that can be shot toward robot.
    env_cfg.scene.obstacle_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleBall",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, 1.0),
            lin_vel=(0.0, 0.0, 0.0),
        ),
    )

    # -------------------- Add contact sensor for ball collision detection -------------------- #
    env_cfg.scene.ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleBall",
        update_period=0.0,
        history_length=4,
        debug_vis=False,
    )

    # -------------------- Add ball observations -------------------- #
    _add_ball_observation_terms(env_cfg.observations.policy)

    # -------------------- Add collision avoidance rewards -------------------- #
    env_cfg.rewards.ball_collision = RewTerm(
        func=mdp.ball_collision_penalty,
        weight=ball_collision_weight,
        params={
            "contact_sensor_cfg": SceneEntityCfg("ball_contact"),
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "threshold": 0.1,
            "min_height_for_ground_filter": ball_collision_min_height_for_ground_filter,
        },
    )

    # Proximity penalty to encourage keeping distance from ball.
    # Training setting:
    # spawn_distance_range = (2.0, 5.0)
    # spawn_height_range   = (0.1, 1.2)
    # ball_speed_range     = (3.0, 6.0)
    env_cfg.rewards.ball_proximity = RewTerm(
        func=mdp.ball_proximity_penalty,
        weight=-5.0,
        params={
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "std": 2.0,
        },
    )
    if ball_approaching_weight is not None:
        env_cfg.rewards.ball_approaching = RewTerm(
            func=mdp.ball_approaching_penalty,
            weight=ball_approaching_weight,
            params={
                "ball_cfg": SceneEntityCfg("obstacle_ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "max_speed": 6.0,
                "danger_distance": 2.0,
            },
        )
    env_cfg.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    spawn_distance_range = (2.0, 5.0)
    spawn_height_range = (0.1, 1.2)
    ball_speed_range = (3.0, 6.0)

    # -------------------- Add ball spawn events -------------------- #
    env_cfg.events.reset_ball = EventTerm(
        func=mdp.reset_ball,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "spawn_distance_range": spawn_distance_range,
            "spawn_height_range": spawn_height_range,
            "ball_speed_range": ball_speed_range,
        },
    )

    env_cfg.events.spawn_ball = EventTerm(
        func=mdp.spawn_ball_towards_robot,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "spawn_distance_range": spawn_distance_range,
            "spawn_height_range": spawn_height_range,
            "ball_speed_range": ball_speed_range,
        },
    )

    # Visual feedback: ball turns red on robot contact.
    env_cfg.events.update_ball_color = EventTerm(
        func=mdp.update_ball_color_on_contact,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "contact_sensor_cfg": SceneEntityCfg("ball_contact"),
            "contact_threshold": 0.1,
            "min_height_for_ground_filter": 0.15,
        },
    )

    # -------------------- Curriculum to gradually enable ball -------------------- #
    env_cfg.curriculum.enable_ball = CurrTerm(
        func=mdp.curriculum_enable_ball,
        params={
            "ball_cfg": SceneEntityCfg("obstacle_ball"),
            "enable_after_iters": enable_ball_after_iters,
            "num_steps_per_env": 24,
        },
    )

    # -------------------- Override base_contact termination -------------------- #
    env_cfg.terminations.base_contact = DoneTerm(
        func=mdp.illegal_contact_unless_ball_enabled,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "min_height_for_ball_hit": 0.4,
        },
    )


# ---------------------------------------------------------------------------- #
#                             Policy Training                                  #
# ---------------------------------------------------------------------------- #

@configclass
class G1CollisionAvoidFlatEnvCfg_POLICY(_G1FlatEnvCfg):
    """
    Environment for training G1 robot to track velocity commands while avoiding ball collisions.
    
    This extends the base G1 flat environment with:
    - A ball rigid object that spawns around the robot and shoots toward it
    - Observations of ball position and velocity relative to robot
    - Rewards/penalties for collision avoidance
    - Contact sensor on robot bodies to detect collisions
    """
    
    def __post_init__(self):
        super().__post_init__()
        _configure_collision_avoid_policy_terms(self)


# ---------------------------------------------------------------------------- #
#                        Recorder Manager Config                               #
# ---------------------------------------------------------------------------- #

from isaaclab.managers import RecorderManagerBaseCfg

@configclass
class G1CollisionAvoidRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager configuration for collision avoidance rollouts."""
    
    # Safety signal: combines collision avoidance with balance signals
    record_safety_signal_collision = mdp.SafetySigCollisionRecorderCfg(
        class_type=mdp.SafetySigCollisionRecorder,
        robot_cfg=SceneEntityCfg("robot"),
        ball_cfg=SceneEntityCfg("obstacle_ball"),
        contact_sensor_cfg=SceneEntityCfg("ball_contact"),
        ball_offset=(0.0, 0.0, 0.8),
        # Collision parameters
        danger_distance=0.3,
        ball_contact_threshold=0.1,
        latch_duration=0.25,
        # Balance parameters (same as flat_push)
        phi_max=3.14159 / 4.0,  # 45 degrees (same as flat_push)
        body_contact_sensor_cfg=SceneEntityCfg("contact_forces", body_names="torso_link"),
        body_contact_threshold=1.0,
        stance_clearance_min=0.25,  # same as flat_push
        # Visualization
        bin_edges=(-0.5, 0.0),
        key_prefix="safety_signal_collision",
    )
    
    # Event: ball spawn (segmenting event)
    record_event_ball_spawn = mdp.EventBallSpawnRecorderCfg(
        class_type=mdp.EventBallSpawnRecorder,
        key_prefix="event_ball_spawn",
    )
    
    # Terminal state
    record_terminal_state = mdp.TerminalStateRecorderCfg(
        class_type=mdp.TerminalStateRecorder,
        key_prefix="terminal_state",
    )
    
    # Policy observations
    record_observations = mdp.ObservationRecorderCfg(
        class_type=mdp.ObservationRecorder,
        key_prefix="policy_obs",
    )


# ---------------------------------------------------------------------------- #
#                            Safety Rollout                                    #
# ---------------------------------------------------------------------------- #

@configclass
class G1CollisionAvoidFlatEnvCfg_SAFETY_ROLLOUT(G1CollisionAvoidFlatEnvCfg_POLICY):
    """
    Environment used to generate policy rollouts for HJ safety value training.
    
    Extends policy training env with:
    - SafetySigCollisionRecorder for computing safety signal
    - EventBallSpawnRecorder for marking segmenting events
    - Ball visualization for safety signal display
    """
    
    recorders: G1CollisionAvoidRecorderManagerCfg = G1CollisionAvoidRecorderManagerCfg()
    
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.enable_ball.params["enable_after_iters"] = 0  # Enable ball from start for safety rollouts

        # -------------------- Safety signal visualization -------------------- #
        viz_safety_signal_collision = "safety_signal_collision"
        self.events.safety_signal_collision_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_signal_collision,
                n_bins=3,  # 3 bins for (-0.5, 0.0) edges: green, yellow, red
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # green to red
                radius=0.07,
            ),
            mode="startup",
        )
        self.events.safety_signal_collision_viz_update = EventTerm(
            func=mdp.ball_viz_update,
            params=dict(
                viz=viz_safety_signal_collision,
                signal_category_attr="safety_signal",
                signal_name="collision",
            ),
            mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        )
        
        # -------------------- Constant command per episode -------------------- #
        # Resample command less frequently for more consistent trajectories
        self.commands.base_velocity.resampling_time_range = (100000000, 100000000)


# ---------------------------------------------------------------------------- #
#                            Safety Inference                                  #
# ---------------------------------------------------------------------------- #

@configclass
class G1CollisionAvoidFlatEnvCfg_SAFETY_INFERENCE(G1CollisionAvoidFlatEnvCfg_SAFETY_ROLLOUT):
    """
    Environment used to verify learned safety value by comparing predictions
    against ground truth safety signal.
    
    Adds SafetyValueInference recorder for running learned model during rollout.
    """
    
    # Path to safety value model directory (set via command-line argument)
    safety_value_path: str = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # -------------------- Safety value predictor -------------------- #
        # Add safety value inference recorder
        if not hasattr(self.recorders, "record_safety_value"):
            self.recorders.record_safety_value = mdp.SafetyValueInferenceCfg(
                class_type=mdp.SafetyValueInference,
                asset=SceneEntityCfg("robot"),
                bin_edges=(-0.5, 0.0),
                key_prefix="safety_value",
                ball_offset=(0.0, 0.0, 1.2),  # Above safety signal ball
            )
        
        # -------------------- Safety value visualization -------------------- #
        viz_safety_value = "safety_value_prediction"
        self.events.safety_value_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_value,
                n_bins=3,  # 3 bins for (-0.5, 0.0) edges
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


# ---------------------------------------------------------------------------- #
#                         29-DoF Unitree Policy Training                       #
# ---------------------------------------------------------------------------- #

@configclass
class G1CollisionAvoidFlat29DofEnvCfg_UNITREE(_G1Flat29DofEnvCfg_UNITREE):
    """
    Collision avoidance env using the Unitree-style 29-DoF flat locomotion base.
    """

    def __post_init__(self):
        super().__post_init__()
        _configure_collision_avoid_policy_terms(self, enable_ball_after_iters=0)

        # Warm-started from a trained/deployed Unitree flat policy, so train avoidance
        # over the final command range instead of replaying the locomotion curriculum.
        if hasattr(self.commands.base_velocity, "limit_ranges"):
            self.commands.base_velocity.ranges = copy.deepcopy(self.commands.base_velocity.limit_ranges)

        # The 29-DoF runner is asymmetric PPO. Keep the actor deployable, but
        # let the critic observe the ball so it can model avoidance rewards.
        if getattr(self.observations, "critic", None) is not None:
            _add_ball_observation_terms(self.observations.critic)


@configclass
class G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK(_G1Flat29DofEnvCfg_UNITREE):
    """
    Forward/backward-only collision avoidance using the Unitree-style 29-DoF flat locomotion base.
    """

    def __post_init__(self):
        super().__post_init__()
        _configure_collision_avoid_policy_terms(
            self,
            enable_ball_after_iters=0,
            ball_collision_weight=-20.0,
            ball_collision_min_height_for_ground_filter=0.15,
            ball_approaching_weight=-2.0,
        )

        # Keep the trained/deployed forward speed range, but remove lateral and yaw commands so
        # avoidance is learned through longitudinal speed changes instead of weak sidestep/turning.
        if hasattr(self.commands.base_velocity, "limit_ranges"):
            self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.ranges = copy.deepcopy(self.commands.base_velocity.limit_ranges)

        if getattr(self.observations, "critic", None) is not None:
            _add_ball_observation_terms(self.observations.critic)


@configclass
class G1CollisionAvoidFlat29DofEnvCfg_UNITREE_SAFETY_ROLLOUT(G1CollisionAvoidFlat29DofEnvCfg_UNITREE):
    """
    Safety rollout env for a Unitree-style 29-DoF collision avoidance policy.
    """

    recorders: G1CollisionAvoidRecorderManagerCfg = G1CollisionAvoidRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()

        self.curriculum.enable_ball.params["enable_after_iters"] = 0

        # -------------------- Safety signal visualization -------------------- #
        viz_safety_signal_collision = "safety_signal_collision"
        self.events.safety_signal_collision_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_signal_collision,
                n_bins=3,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                radius=0.07,
            ),
            mode="startup",
        )
        self.events.safety_signal_collision_viz_update = EventTerm(
            func=mdp.ball_viz_update,
            params=dict(
                viz=viz_safety_signal_collision,
                signal_category_attr="safety_signal",
                signal_name="collision",
            ),
            mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        )

        # Match the final Unitree command curriculum range used by the deployed controller.
        if hasattr(self.commands.base_velocity, "limit_ranges"):
            self.commands.base_velocity.ranges = copy.deepcopy(self.commands.base_velocity.limit_ranges)

        # -------------------- Constant command per episode -------------------- #
        self.commands.base_velocity.resampling_time_range = (100000000, 100000000)


@configclass
class G1CollisionAvoidFlat29DofEnvCfg_UNITREE_SAFETY_INFERENCE(
    G1CollisionAvoidFlat29DofEnvCfg_UNITREE_SAFETY_ROLLOUT
):
    """
    Safety inference env for a Unitree-style 29-DoF collision avoidance policy.
    """

    # Path to safety value model directory (set via command-line argument)
    safety_value_path: str = None

    def __post_init__(self):
        super().__post_init__()

        if not hasattr(self.recorders, "record_safety_value"):
            self.recorders.record_safety_value = mdp.SafetyValueInferenceCfg(
                class_type=mdp.SafetyValueInference,
                asset=SceneEntityCfg("robot"),
                bin_edges=(-0.5, 0.0),
                key_prefix="safety_value",
                ball_offset=(0.0, 0.0, 1.2),
            )

        # -------------------- Safety value visualization -------------------- #
        viz_safety_value = "safety_value_prediction"
        self.events.safety_value_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_value,
                n_bins=3,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
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
class G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK_SAFETY_ROLLOUT(
    G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK
):
    """
    Safety rollout env for the forward/backward Unitree-style 29-DoF collision avoidance policy.
    """

    recorders: G1CollisionAvoidRecorderManagerCfg = G1CollisionAvoidRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()

        self.curriculum.enable_ball.params["enable_after_iters"] = 0

        # -------------------- Safety signal visualization -------------------- #
        viz_safety_signal_collision = "safety_signal_collision"
        self.events.safety_signal_collision_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_signal_collision,
                n_bins=3,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                radius=0.07,
            ),
            mode="startup",
        )
        self.events.safety_signal_collision_viz_update = EventTerm(
            func=mdp.ball_viz_update,
            params=dict(
                viz=viz_safety_signal_collision,
                signal_category_attr="safety_signal",
                signal_name="collision",
            ),
            mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
        )

        # Keep the same forward/backward command distribution for safety data collection.
        if hasattr(self.commands.base_velocity, "limit_ranges"):
            self.commands.base_velocity.ranges = copy.deepcopy(self.commands.base_velocity.limit_ranges)

        # -------------------- Constant command per episode -------------------- #
        self.commands.base_velocity.resampling_time_range = (100000000, 100000000)


@configclass
class G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK_SAFETY_INFERENCE(
    G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK_SAFETY_ROLLOUT
):
    """
    Safety inference env for the forward/backward Unitree-style 29-DoF collision avoidance policy.
    """

    # Path to safety value model directory (set via command-line argument)
    safety_value_path: str = None

    def __post_init__(self):
        super().__post_init__()

        if not hasattr(self.recorders, "record_safety_value"):
            self.recorders.record_safety_value = mdp.SafetyValueInferenceCfg(
                class_type=mdp.SafetyValueInference,
                asset=SceneEntityCfg("robot"),
                bin_edges=(-0.5, 0.0),
                key_prefix="safety_value",
                ball_offset=(0.0, 0.0, 1.2),
            )

        # -------------------- Safety value visualization -------------------- #
        viz_safety_value = "safety_value_prediction"
        self.events.safety_value_viz_setup = EventTerm(
            func=mdp.ball_viz_setup,
            params=dict(
                viz=viz_safety_value,
                n_bins=3,
                color_range=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
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
