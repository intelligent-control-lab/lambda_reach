import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# Register a new env matching the official Isaac-Velocity-Flat-G1-v0
# using locally derived cfg classes for future overrides.
gym.register(
    id="Isaac-Velocity-Flat-G1-PPO-POLICY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_POLICY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-PPO-SAFETY-ROLLOUT",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_SAFETY_ROLLOUT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-PPO-SAFETY-INFERENCE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_SAFETY_INFERENCE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


# Register the standalone Unitree-style 29-DoF G1 flat task.
gym.register(
    id="Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO-POLICY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1Flat29DofEnvCfg_UNITREE",
        "play_env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1Flat29DofEnvCfg_UNITREE_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Flat29DofUnitreeAsymPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO-SAFETY-ROLLOUT",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1Flat29DofEnvCfg_UNITREE_SAFETY_ROLLOUT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Flat29DofUnitreeAsymPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO-SAFETY-INFERENCE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1Flat29DofEnvCfg_UNITREE_SAFETY_INFERENCE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Flat29DofUnitreeAsymPPORunnerCfg",
    },
)


# Register a new env matching the official Isaac-Velocity-Rough-G1-v0
# using locally derived cfg classes for future overrides.
gym.register(
    id="Isaac-Velocity-Rough-G1-PPO-POLICY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_POLICY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-PPO-SAFETY-ROLLOUT",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_SAFETY_ROLLOUT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-PPO-SAFETY-INFERENCE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_SAFETY_INFERENCE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
