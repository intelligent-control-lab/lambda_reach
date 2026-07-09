# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

#
# Derived PPO cfg aligned with official G1 Flat task, for future overrides.
#
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg as _G1FlatPPORunnerCfg,
    G1RoughPPORunnerCfg as _G1RoughPPORunnerCfg,
)


@configclass
class G1FlatPPORunnerCfg(_G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
    
        self.experiment_name = "g1_flat_ppo"


@configclass
class G1Flat29DofUnitreeAsymPPORunnerCfg(_G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_29dof_flat_unitree_ppo"
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.max_iterations = 50000
        self.save_interval = 100
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.01


@configclass
class G1RoughPPORunnerCfg(_G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_rough_ppo"
