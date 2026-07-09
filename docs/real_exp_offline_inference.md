# Offline Real-Experiment Data Loading and Model Inference

This document explains how to load the real-robot experiment logs under `logs/real_exp`, and how to run offline inference on those trajectories with the base policy checkpoint and safe value checkpoint used for deployment.

Script:

```text
scripts/real_exp_offline_inference.py
```

## 1. Data Location and Directory Layout

The real-robot experiment data has been copied to:

```text
/home/shangtao/project/hj_humanoid/logs/real_exp
```

Each experiment is stored as one run directory. A typical layout is:

```text
logs/real_exp/
  2026-05-10_22-07-06_velocity/
    metadata.yaml
    records.csv
    context_raw.csv

  demo_push_1/
    2026-05-10_22-36-30_velocity/
      metadata.yaml
      records.csv
      context_raw.csv
```

The script recursively searches for all `records.csv` files, so it automatically loads both top-level runs and nested runs under directories such as `demo_push_*` and `test_without_support`.

## 2. File Contents

### metadata.yaml

`metadata.yaml` stores experiment metadata and deployment model paths, for example:

```yaml
robot: g1_29dof
fsm_state: Velocity
recording_scope: velocity_with_context
control_dt: 0.02
context_seconds: 5
context_sample_dt: 0.02
policy_model: /home/shangtao/project/unitree_rl_lab/.../exported/policy.onnx
safety_value_model: /home/shangtao/project/unitree_rl_lab/.../exported/safety_value.onnx
obs_dim: 480
action_dim: 29
safety_signal: tilt_only
```

### records.csv

`records.csv` is the main file. Each row corresponds to one velocity-control step, with a nominal control period of `0.02s`.

Important fields:

```text
time_s, relative_time_s, control_step, dt_actual
root_ang_vel_0..2
root_quat_0..3
projected_gravity_0..2
joint_pos_0..28
joint_vel_0..28
command_0..2
obs_key
obs_0..479
safety_valid
safety_value
safety_signal
safety_inference_ms
prev_action_0..28
policy_action_0..28
target_q_0..28
policy_inference_ms
```

The most important fields are:

```text
obs_0..obs_479              input to the policy and safe value networks
policy_action_0..28         base policy action logged during real deployment
target_q_0..28              PD target joint positions after action scale/offset
safety_value                safe value network output logged during real deployment
safety_signal               tilt-only safety signal
```

### context_raw.csv

`context_raw.csv` stores raw state samples before and after the velocity run. It is useful for analyzing state before entering or after leaving Velocity mode, but it is not required for policy inference.

Fields:

```text
time_s, relative_time_s, phase, fsm_state
root_ang_vel_0..2
root_quat_0..3
projected_gravity_0..2
joint_pos_0..28
joint_vel_0..28
```

`phase` can be:

```text
pre_velocity
post_velocity
```

## 3. Observation Format

`records.csv` already stores the exact observation vectors that were fed into the deployed networks. Offline inference does not need to reconstruct observations from raw state.

Use these columns directly:

```text
obs = records[obs_0..obs_479]
```

The observation dimension is `480`. The concatenation order is:

```text
base_ang_vel                 3 * 5   -> obs_0..14
projected_gravity            3 * 5   -> obs_15..29
keyboard_velocity_commands   3 * 5   -> obs_30..44
joint_pos_rel               29 * 5   -> obs_45..189
joint_vel_rel               29 * 5   -> obs_190..334
last_action                 29 * 5   -> obs_335..479
```

Each term has `history_length=5`. For example, the current command is the last frame of the command history:

```text
obs_42, obs_43, obs_44
```

## 4. Conda Environment

Use `env_isaaclab`:

```bash
/home/shangtao/miniconda3/envs/env_isaaclab/bin/python scripts/real_exp_offline_inference.py --help
```

Note: `env_isaaclab` currently does not have `onnxruntime`. In default `auto` mode, the script skips the ONNX paths recorded in `metadata.yaml` and falls back to the matching PyTorch checkpoints in this repo.

## 5. Default Checkpoints

By default, the script uses the following PyTorch checkpoints:

### Base policy

```text
logs/rsl_rl/g1_29dof_flat_unitree_ppo/2026-04-24_13-47-26/model_6000.pt
```

This is the raw RSL-RL actor checkpoint. The script reads:

```text
actor_state_dict
```

and builds the actor MLP for deterministic inference.

### Safe value

```text
logs/safety_analysis/g1_29dof_flat_unitree_ppo_6000/results/lambda_reachability_lambda_reach_lambda_0_99/models/last_seed_0.pt
```

This is the DPE / lambda reachability safe value checkpoint. The script reads:

```text
critic1
critic2
```

and uses the same logic as the ONNX export:

```text
safety_value = 0.5 * (critic1(obs) + critic2(obs))
```

## 6. Run All Real-Experiment Data

Run this from the repo root:

```bash
cd /home/shangtao/project/hj_humanoid

/home/shangtao/miniconda3/envs/env_isaaclab/bin/python scripts/real_exp_offline_inference.py
```

Default input:

```text
logs/real_exp
```

Default output:

```text
logs/real_exp_offline_inference
```

Output layout:

```text
logs/real_exp_offline_inference/
  summary.csv
  2026-05-10_22-07-06_velocity/
    predictions.npz
  demo_push_1__2026-05-10_22-36-30_velocity/
    predictions.npz
  ...
```

## 7. Output Interpretation

### summary.csv

Each row in `summary.csv` corresponds to one run and contains:

```text
run
num_rows
duration_s
policy_mae
policy_rmse
policy_max_abs
safety_value_mae
safety_value_rmse
safety_value_max_abs
mean_logged_safety_value
min_logged_safety_value
max_logged_safety_value
mean_safety_signal
max_safety_signal
```

where:

```text
policy_mae = mean(abs(policy_action_pred - policy_action_logged))
safety_value_mae = mean(abs(safety_value_pred - safety_value_logged))
```

If the checkpoints match the deployed models, the errors should be at floating-point noise level. With `env_isaaclab`, the verified small-sample error is around `1e-7`.

### predictions.npz

Each run's `predictions.npz` contains:

```text
time_s
relative_time_s
control_step
command
policy_action_logged
policy_action_pred
target_q_logged
prev_action_logged
safety_value_logged
safety_value_pred
safety_signal
```

Load it like this:

```python
import numpy as np

data = np.load("logs/real_exp_offline_inference/2026-05-10_22-07-06_velocity/predictions.npz")

policy_action_logged = data["policy_action_logged"]  # [N, 29]
policy_action_pred = data["policy_action_pred"]      # [N, 29]
safety_value_logged = data["safety_value_logged"]    # [N]
safety_value_pred = data["safety_value_pred"]        # [N]
safety_signal = data["safety_signal"]                # [N]
```

By default, `obs` is not saved to avoid large output files. To save `obs` as well:

```bash
/home/shangtao/miniconda3/envs/env_isaaclab/bin/python scripts/real_exp_offline_inference.py --save-obs
```

## 8. Process a Single Run

Use `--run-filter`:

```bash
/home/shangtao/miniconda3/envs/env_isaaclab/bin/python scripts/real_exp_offline_inference.py \
  --run-filter 2026-05-10_22-07-06_velocity
```

For a quick smoke test, limit the number of rows per run:

```bash
/home/shangtao/miniconda3/envs/env_isaaclab/bin/python scripts/real_exp_offline_inference.py \
  --max-rows 20 \
  --no-save-predictions
```

## 9. Explicitly Specify Checkpoints

To explicitly specify model paths:

```bash
/home/shangtao/miniconda3/envs/env_isaaclab/bin/python scripts/real_exp_offline_inference.py \
  --policy-model /home/shangtao/project/hj_humanoid/logs/rsl_rl/g1_29dof_flat_unitree_ppo/2026-04-24_13-47-26/model_6000.pt \
  --safety-value-model /home/shangtao/project/hj_humanoid/logs/safety_analysis/g1_29dof_flat_unitree_ppo_6000/results/lambda_reachability_lambda_reach_lambda_0_99/models/last_seed_0.pt
```

If you use a DPE checkpoint but `training_config.json` is not in the checkpoint's expected parent directory, pass:

```bash
--safety-value-dir /path/to/safety_value_result_dir
```

That directory should contain:

```text
training_config.json
models/last_seed_0.pt
```

## 10. Use ONNX Models

The real deployment metadata records ONNX model paths:

```text
policy.onnx
safety_value.onnx
```

If the current Python environment has `onnxruntime`, you can explicitly use the ONNX models:

```bash
/home/shangtao/miniconda3/envs/env_isaaclab/bin/python scripts/real_exp_offline_inference.py \
  --policy-model /home/shangtao/project/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/unitree_flat_baseline_v4/exported/policy.onnx \
  --safety-value-model /home/shangtao/project/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/unitree_flat_baseline_v4/exported/safety_value.onnx
```

If `onnxruntime` is not installed, explicitly passing ONNX paths will fail. In that case, use the default command and the script will automatically use the `.pt` checkpoints.

## 11. Direct Python Usage

You can also import the loader and model wrappers directly:

```python
from pathlib import Path

from scripts.real_exp_offline_inference import (
    load_policy_model,
    load_real_exp_records,
    load_safety_value_model,
)

records = load_real_exp_records(
    Path("logs/real_exp/2026-05-10_22-07-06_velocity/records.csv")
)

obs = records.obs  # [N, 480], float32

policy = load_policy_model(
    Path("logs/rsl_rl/g1_29dof_flat_unitree_ppo/2026-04-24_13-47-26/model_6000.pt"),
    device="cpu",
)
safety_value = load_safety_value_model(
    Path("logs/safety_analysis/g1_29dof_flat_unitree_ppo_6000/results/lambda_reachability_lambda_reach_lambda_0_99/models/last_seed_0.pt"),
    device="cpu",
)

policy_action_pred = policy.predict(obs)          # [N, 29]
safety_value_pred = safety_value.predict(obs)     # [N, 1]
```

Compare against the outputs logged during real deployment:

```python
policy_action_logged = records.policy_action      # [N, 29]
safety_value_logged = records.safety_value        # [N]

policy_error = abs(policy_action_pred - policy_action_logged).mean()
safety_error = abs(safety_value_pred[:, 0] - safety_value_logged).mean()

print(policy_error, safety_error)
```

## 12. FAQ

### 1. Why does the default mode not use the ONNX paths from metadata?

Because the current `env_isaaclab` environment does not have `onnxruntime`. In `auto` mode, the script checks the environment and uses the matching `.pt` checkpoints from this repo if ONNX cannot be loaded.

### 2. Why should `safety_value` not be treated as ground truth?

The `safety_value` column in `records.csv` is the prediction from the deployed safe value network, not a ground-truth label. It is suitable for checking whether a checkpoint matches the real deployment output, or for distilling an old model. If you want to retrain a safe value model, define a separate target.

### 3. When do we need `context_raw.csv`?

For policy / safe value inference on real trajectories, use `obs_0..479` from `records.csv`.

Use `context_raw.csv` only when you need to:

```text
analyze raw state before entering or after leaving Velocity mode
reconstruct observations from raw state
regenerate observations after changing the observation layout
analyze raw IMU / joint-state signals
```
