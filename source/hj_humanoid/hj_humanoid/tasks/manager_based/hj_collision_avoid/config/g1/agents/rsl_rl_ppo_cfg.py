# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for G1 collision avoidance task."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg as _G1FlatPPORunnerCfg,
)
from hj_humanoid.tasks.manager_based.hj_flat_push.config.g1.agents.rsl_rl_ppo_cfg import (
    G1Flat29DofUnitreeAsymPPORunnerCfg as _G1Flat29DofUnitreeAsymPPORunnerCfg,
)


@configclass
class G1CollisionAvoidFlatPPORunnerCfg(_G1FlatPPORunnerCfg):
    """PPO runner configuration for G1 collision avoidance on flat terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.experiment_name = "g1_collision_avoid_flat_ppo"
        self.max_iterations = 1500


@configclass
class G1CollisionAvoidFlat29DofUnitreeAsymPPORunnerCfg(_G1Flat29DofUnitreeAsymPPORunnerCfg):
    """Asymmetric PPO runner for 29-DoF Unitree-style G1 collision avoidance."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_collision_avoid_flat_29dof_unitree_ppo"
        self.max_iterations = 5000


@configclass
class G1CollisionAvoidFlat29DofUnitreeFwdBackAsymPPORunnerCfg(_G1Flat29DofUnitreeAsymPPORunnerCfg):
    """Asymmetric PPO runner for forward/backward 29-DoF Unitree-style G1 collision avoidance."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo"
        self.max_iterations = 5000
