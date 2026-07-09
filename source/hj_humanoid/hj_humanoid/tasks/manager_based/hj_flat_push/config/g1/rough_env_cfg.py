# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.utils import configclass

import hj_humanoid.tasks.manager_based.hj_flat_push.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg as _G1RoughEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

# ---------------------------------------------------------------------------- #
#                                 Policy Train                                 #
# ---------------------------------------------------------------------------- #

@configclass
class G1RoughEnvCfg_POLICY(_G1RoughEnvCfg):
    """
    Env used to train the RL policy to be certified on rough terrain.
    """

    pass

# ---------------------------------------------------------------------------- #
#                                   HJ Learn                                   #
# ---------------------------------------------------------------------------- #

@configclass
class G1RoughEnvRecorderManagerCfg(RecorderManagerBaseCfg):

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
class G1RoughEnvCfg_SAFETY_ROLLOUT(G1RoughEnvCfg_POLICY):
    """
    Env used to generate policy rollouts for HJ training on rough terrain.
    """

    recorders: G1RoughEnvRecorderManagerCfg = G1RoughEnvRecorderManagerCfg()

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

# ---------------------------------------------------------------------------- #
#                                    HJ Play                                   #
# ---------------------------------------------------------------------------- #
@configclass
class G1RoughEnvCfg_SAFETY_INFERENCE(G1RoughEnvCfg_SAFETY_ROLLOUT):
    """
    Env used to verify safety value by comparing safety sign and max l value in finite look-ahead (normally) on rough terrain.
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
