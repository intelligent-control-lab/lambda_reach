import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Collision Avoidance Flat Environment - POLICY training
gym.register(
    id="Isaac-Collision-Avoid-Flat-G1-PPO-POLICY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1CollisionAvoidFlatEnvCfg_POLICY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CollisionAvoidFlatPPORunnerCfg",
    },
)

# Collision Avoidance Flat Environment - SAFETY ROLLOUT (for HJ data collection)
gym.register(
    id="Isaac-Collision-Avoid-Flat-G1-PPO-SAFETY-ROLLOUT",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1CollisionAvoidFlatEnvCfg_SAFETY_ROLLOUT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CollisionAvoidFlatPPORunnerCfg",
    },
)

# Collision Avoidance Flat Environment - SAFETY INFERENCE (for verifying learned safety value)
gym.register(
    id="Isaac-Collision-Avoid-Flat-G1-PPO-SAFETY-INFERENCE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1CollisionAvoidFlatEnvCfg_SAFETY_INFERENCE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CollisionAvoidFlatPPORunnerCfg",
    },
)

# Collision Avoidance Flat Environment - 29-DoF Unitree-style forward/backward POLICY training
gym.register(
    id="Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO-POLICY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1CollisionAvoidFlat29DofUnitreeFwdBackAsymPPORunnerCfg"
        ),
    },
)

# Collision Avoidance Flat Environment - 29-DoF Unitree-style forward/backward SAFETY ROLLOUT
gym.register(
    id="Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO-SAFETY-ROLLOUT",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK_SAFETY_ROLLOUT"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1CollisionAvoidFlat29DofUnitreeFwdBackAsymPPORunnerCfg"
        ),
    },
)

# Collision Avoidance Flat Environment - 29-DoF Unitree-style forward/backward SAFETY INFERENCE
gym.register(
    id="Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO-SAFETY-INFERENCE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK_SAFETY_INFERENCE"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1CollisionAvoidFlat29DofUnitreeFwdBackAsymPPORunnerCfg"
        ),
    },
)
