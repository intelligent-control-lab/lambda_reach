# Unitree G1 Collision Avoidance Experiment

This note documents the real-robot collision avoidance experiment pipeline for the 29-DoF Unitree G1.

The avoidance experiment trains a locomotion policy that reacts to a moving ball tracked by an OptiTrack MoCap system, alongside a safety value function that predicts future collision or fall risk.

## 1. Train the Avoidance Policy

The avoidance task is registered as:

```text
Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO
```

The training script appends `-POLICY` internally; the Gym task actually registered is:

```text
Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO-POLICY
```

Relevant source files:

```text
safety_value/scripts/0_policy/train.py
source/hj_humanoid/hj_humanoid/tasks/manager_based/hj_collision_avoid/config/g1/__init__.py
source/hj_humanoid/hj_humanoid/tasks/manager_based/hj_collision_avoid/config/g1/agents/rsl_rl_ppo_cfg.py
source/hj_humanoid/hj_humanoid/tasks/manager_based/hj_collision_avoid/config/g1/flat_env_cfg.py
```

### Environment Setup

Run from the repository root inside the Isaac Lab Python environment:

```bash
python -m pip install -e source/hj_humanoid
python -m pip install -e safety_value
```

### Training Command

```bash
python safety_value/scripts/0_policy/train.py \
  --task Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42 \
  --headless
```

### Output Layout

```text
experiment_name = g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo
```

Training outputs are written to:

```text
logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/<timestamp>/
```

Choose one checkpoint for deployment and keep it fixed for all downstream steps:

```text
logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/<POLICY_RUN>/model_<POLICY_STEP>.pt
```

### What This Policy Trains

```text
actor observation: 510 dims
action:            29 dims
control rate:      50 Hz
```

The 510-dimensional actor observation is a 5-frame history of:

```text
base_ang_vel          3   (scale ×0.2)
projected_gravity     3
velocity_commands     3
joint_pos_rel        29
joint_vel_rel        29   (scale ×0.05)
last_action          29
ball_position         3   (in robot base frame, from MoCap)
ball_velocity         3   (in robot base frame, from MoCap)
```

That is `96 + 6 = 102` dims per frame, stacked for 5 frames.
The ball position and velocity are provided by the MoCap pipeline described in step 5.

## 2. Train the Safety Value Function

The safety value function is trained from Isaac Lab rollouts of the avoidance policy.
It is not trained from real-robot logs.

Set shell variables for the chosen policy checkpoint:

```bash
POLICY_RUN=<your_policy_run>
POLICY_STEP=<your_policy_step>
POLICY_CKPT=logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/${POLICY_RUN}/model_${POLICY_STEP}.pt
SA_ROOT=g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo_${POLICY_STEP}
```

Relevant source files:

```text
safety_value/scripts/1_data_prep/rollout_safety_data.py
safety_value/scripts/1_data_prep/data_processing.py
safety_value/scripts/2_safety_analysis/train.py
```

### 2.1 Roll Out Safety Data

```bash
python safety_value/scripts/1_data_prep/rollout_safety_data.py \
  --task Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO \
  --checkpoint ${POLICY_CKPT} \
  --safety_analysis_root ${SA_ROOT} \
  --num_envs 2048 \
  --video_length 1000 \
  --seed 42 \
  --headless
```

Output:

```text
logs/safety_analysis/<SA_ROOT>/data_raw.hdf5
```

### 2.2 Process the Rollout Dataset

```bash
python safety_value/scripts/1_data_prep/data_processing.py \
  --safety_analysis_root logs/safety_analysis/${SA_ROOT} \
  --raw_dataset_name data_raw.hdf5 \
  --test_proportion 0.2
```

Output:

```text
logs/safety_analysis/<SA_ROOT>/data_processed_train.pkl
logs/safety_analysis/<SA_ROOT>/data_processed_test.pkl
```

### 2.3 Train Lambda-Reachability

```bash
python safety_value/scripts/2_safety_analysis/train.py \
  --safety_analysis_root ${SA_ROOT} \
  --algo lambda_reachability \
  --model_tag lambda_reachability \
  --run_name lambda_reach_lambda_0_99 \
  --lambda_param 0.99 \
  --max_horizon 200 \
  --alpha_bce 5.0 \
  --weight_main 1.0 \
  --weight_hinge_lb 1.0 \
  --weight_hinge_mono 1.0 \
  --weight_bce 1.0 \
  --target_tau_lambda 0.005 \
  --target_update_period 1 \
  --total_steps 10000 \
  --eval_steps 500 \
  --eval_batches 0 \
  --batch 256 \
  --lr 1e-3 \
  --hidden_dims 256 256 \
  --seed 0 \
  --device cuda \
  --epochs 20
```

Output:

```text
logs/safety_analysis/<SA_ROOT>/results/
  lambda_reachability_lambda_reach_lambda_0_99/
    training_config.json
    models/last_seed_0.pt
```

## 3. Export Models to ONNX

```bash
POLICY_RUN=<your_policy_run>
POLICY_STEP=<your_policy_step>
POLICY_CKPT=logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/${POLICY_RUN}/model_${POLICY_STEP}.pt
SA_ROOT=g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo_${POLICY_STEP}
SV_RUN=lambda_reachability_lambda_reach_lambda_0_99
SV_DIR=logs/safety_analysis/${SA_ROOT}/results/${SV_RUN}
```

### 3.1 Export the Avoidance Policy

```bash
python safety_value/scripts/0_policy/play.py \
  --task Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO \
  --checkpoint ${POLICY_CKPT} \
  --num_envs 1 \
  --headless
```

Output:

```text
logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/<POLICY_RUN>/exported/policy.onnx
```

The ONNX actor takes a `[1, 510]` observation and produces a `[1, 29]` action.

### 3.2 Export the Safety Value Function

```bash
python deploy/tools/export_onnx.py \
  --safety_value_dir ${SV_DIR} \
  --output ${SV_DIR}/exported/safety_value.onnx
```

The safety value ONNX takes a `[1, 510]` observation and outputs a `[1, 1]` scalar.

## 4. Set Up the MoCap Pipeline

The avoidance policy requires ball position and velocity in the robot base frame at 50 Hz.
This is provided by a two-node ROS 2 pipeline:

```text
OptiTrack/Motive
  → natnet_ros2       (ROS 2, publishes rigid body poses)
  → mocap_avoid_bridge (ROS 2, computes relative ball obs → UDP)
  → unitree_rl_lab MocapClient (C++, port 15151)
```

### 4.1 Install natnet_ros2

Clone the upstream driver into the workspace source directory:

```bash
cd deploy/avoid/ros2_ws/src
git clone https://github.com/L2S-lab/natnet_ros2
```

Follow `natnet_ros2/README.md` to install the NatNet SDK and configure your OptiTrack rigid bodies.
The bridge expects two rigid bodies published as `PoseStamped` topics:

```text
/g1_base/pose      — rigid body attached to the robot base
/avoid_ball/pose   — rigid body on the ball
```

### 4.2 Build and Launch the Bridge

```bash
cd deploy/avoid/ros2_ws
colcon build --packages-select mocap_avoid_bridge natnet_ros2
source install/setup.bash
```

Before launching, open `deploy/avoid/ros2_ws/src/mocap_avoid_bridge/config/mocap_avoid_bridge.yaml`
and set `udp_enabled: true` to enable the UDP output to `unitree_rl_lab`:

```yaml
udp_enabled: true
udp_host: 127.0.0.1
udp_port: 15151
```

Then launch:

```bash
ros2 launch mocap_avoid_bridge mocap_avoid_bridge.launch.py
```

Verify the bridge is receiving valid poses — it logs a warning if either rigid body pose is stale or the two timestamps are out of sync.

## 5. Deploy with unitree_rl_lab

Clone [unitree_rl_lab](https://github.com/intelligent-control-lab/unitree_rl_lab-lambda) and set:

```bash
UNITREE_RL_LAB=<path-to-unitree_rl_lab>
```

### 5.1 Create a Policy Directory

```bash
POLICY_NAME=my_avoid_policy
DEPLOY_DIR=${UNITREE_RL_LAB}/deploy/robots/g1_29dof/config/policy/velocity/${POLICY_NAME}

mkdir -p ${DEPLOY_DIR}/exported ${DEPLOY_DIR}/params

cp logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/${POLICY_RUN}/exported/policy.onnx \
   ${DEPLOY_DIR}/exported/policy.onnx
cp ${SV_DIR}/exported/safety_value.onnx \
   ${DEPLOY_DIR}/exported/safety_value.onnx
```

Create `${DEPLOY_DIR}/params/deploy.yaml` based on the reference avoid policy
(`avoid/params/deploy.yaml` in the repo). Key differences from the push policy:

```text
obs input dim:     510 (adds ball_position×5 + ball_velocity×5)
lin_vel_x range:   [-0.5, 1.0]
```

### 5.2 Point config.yaml at the New Policy

Edit `${UNITREE_RL_LAB}/deploy/robots/g1_29dof/config/config.yaml`:

```yaml
  Velocity:
    policy_dir: config/policy/velocity/my_avoid_policy
    safety_value:
      enabled: true
      model_path: config/policy/velocity/my_avoid_policy/exported/safety_value.onnx
      log_dir: log/safety_value
      log_obs: true
      log_state: true
      safety_signal_type: mocap_collision_avoid
      avoid_signal_danger_distance: 0.3
      avoid_signal_contact_distance: 0.3
      avoid_signal_latch_duration_s: 0.25
    mocap_avoid:
      enabled: true
      port: 15151
      max_age_s: 0.10
```

`safety_signal_type: mocap_collision_avoid` uses both ball proximity and tilt to compute
the safety signal, matching the avoid task training setup.

### 5.3 Build the Controller

```bash
cd ${UNITREE_RL_LAB}/deploy/robots/g1_29dof
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### 5.4 Real-Robot Deployment

Ensure the MoCap pipeline (step 4) is running and the bridge is publishing valid observations
before starting the controller.

```bash
cd ${UNITREE_RL_LAB}/deploy/robots/g1_29dof/build
./g1_ctrl --network eth0   # replace eth0 with your interface name
```

FSM transitions via keyboard (`f` → FixStand, `v` → Velocity, `p` → Passive) or joystick
(`LT + Up` / `RB + X` / `LT + B`).

The safety-value monitor logs observations and scores to `log/safety_value/` for post-hoc
analysis with `deploy/tools/safety_value_dashboard.py` (live) or
`deploy/tools/real_exp_offline_inference.py` (post-hoc).
