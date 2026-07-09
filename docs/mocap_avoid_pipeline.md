# MoCap Avoid Pipeline

This document describes the real-time pipeline for feeding OptiTrack/Motive rigid-body data into the 29-DoF G1 collision-avoidance policy.

## Goal

The avoidance actor expects the normal 480-dimensional Unitree flat observation plus 30 additional dimensions:

```text
obs_480..494  ball_position_in_base_frame history, 3 * 5
obs_495..509  ball_velocity_in_base_frame history, 3 * 5
```

The MoCap bridge publishes the current 6-dimensional value:

```text
[ball_pos_base_x, ball_pos_base_y, ball_pos_base_z,
 ball_vel_base_x, ball_vel_base_y, ball_vel_base_z]
```

The existing Unitree deploy `ObservationManager` should keep the 5-frame history from those current 3D terms.

## Architecture

```text
Motive / NatNet
  -> NatNet ROS2 driver
  -> /g1_base/pose, /avoid_ball/pose

mocap_avoid_bridge ROS2 node
  -> subscribes both PoseStamped topics
  -> estimates base and ball linear velocity by filtered finite difference
  -> computes ball position and velocity in the G1 base frame
  -> publishes /avoid/mocap_observation and /avoid/mocap_valid
  -> optionally sends a compact UDP packet to a non-ROS controller process

unitree_rl_lab g1_ctrl
  -> reads the latest 6D observation asynchronously
  -> exposes ball_position and ball_velocity observation terms
  -> runs the 510D avoid policy and 510D avoid safety value
  -> still outputs the same 29-DoF joint action
```

## ROS2 Bridge

Package location:

```text
deploy/avoid/ros2_ws/src/mocap_avoid_bridge
```

Default launch:

```bash
cd deploy/avoid/ros2_ws
colcon build --packages-select mocap_avoid_bridge
source install/setup.bash
ros2 launch mocap_avoid_bridge mocap_avoid_bridge.launch.py
```

Default parameters:

```text
base_pose_topic: /g1_base/pose
ball_pose_topic: /avoid_ball/pose
output_topic: /avoid/mocap_observation
valid_topic: /avoid/mocap_valid
output_rate_hz: 50.0
max_pose_age_s: 0.10
max_stamp_delta_s: 0.04
velocity_filter_alpha: 0.35
udp_enabled: false
```

The output topic is `std_msgs/Float32MultiArray` with six values:

```text
data[0:3] = ball_pos_base
data[3:6] = ball_vel_base
```

The valid topic is `std_msgs/Bool`. Consumers must treat `false` as a safety-critical stale/missing MoCap state.

## Coordinate Computation

The bridge treats NatNet poses as world-frame poses.

```text
rel_pos_w = ball_pos_w - base_pos_w
rel_vel_w = ball_vel_w - base_vel_w

ball_pos_base = R_base_w^T * rel_pos_w
ball_vel_base = R_base_w^T * rel_vel_w
```

This matches the Isaac Lab training observation functions:

```text
ball_position_in_robot_frame
ball_velocity_in_robot_frame
```

The ball orientation is not used by the policy. The ball rigid-body pivot should still be calibrated to the ball center in Motive.

## Real-Time Design

The system has three different real-time-ish loops:

```text
NatNet/ROS2 callbacks       pose producer, often 100-240 Hz, network-dependent
Policy thread               env->step_dt = 0.02, about 50 Hz
Unitree command FSM thread  about 1 kHz
```

The ROS2 callback path must not block the robot command loop. The bridge therefore follows a producer-consumer model:

```text
ROS2 callbacks:
  receive PoseStamped
  update latest pose and filtered velocity under a short lock

50 Hz bridge timer:
  copies latest base and ball states
  validates freshness and timestamp alignment
  computes the 6D observation
  publishes ROS2 output and optional UDP output

Unitree controller:
  should only read the latest available 6D value
  should never wait for a ROS message inside an observation function
```

If MoCap is stale or unsynchronized, the controller should stop using the avoid policy or transition to a safe state. It should not silently replace the ball observation with zeros.

## Optional UDP Packet

For `g1_ctrl`, a plain UDP handoff is the smallest integration surface because the controller is currently a plain CMake + Unitree SDK process, not an ament/rclcpp ROS2 node.

Enable it in `config/mocap_avoid_bridge.yaml`:

```yaml
udp_enabled: true
udp_host: 127.0.0.1
udp_port: 15151
```

Binary packet format, little endian:

```text
magic[4]       = "MAVO"
version:uint16 = 1
size:uint16    = packet size in bytes
seq:uint64
stamp_s:double
valid:uint8
padding[3]
ball_pos_base:float32[3]
ball_vel_base:float32[3]
```

On invalid MoCap, the bridge keeps sending packets with `valid=0` and the last known observation values. The controller must check `valid` and packet age.

## Unitree Integration Plan

The existing `unitree_rl_lab` controller is integrated in a narrow, reversible way. The default flat deploy chain remains unchanged:

```text
deploy/robots/g1_29dof/config/config.yaml
  FSM.Velocity.policy_dir: config/policy/velocity/unitree_flat_baseline_v4
```

The MoCap avoid path is staged separately:

```text
/home/shangtao/project/unitree_rl_lab/deploy/include/isaaclab/algorithms/mocap_avoid_client.h
/home/shangtao/project/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/unitree_avoid_fwd_back_mocap/
```

The added Unitree-side pieces are:

1. Add a small async `MocapAvoidClient`.
   It receives the UDP packet in a background thread and stores the latest packet in a thread-safe cache.

2. Register two new observation functions:

   ```cpp
   REGISTER_OBSERVATION(ball_position)
   REGISTER_OBSERVATION(ball_velocity)
   ```

   Each function should only read the latest cached 3D vector. It must not block.

3. Add a new avoid policy directory:

   ```text
   config/policy/velocity/unitree_avoid_fwd_back/
     params/deploy.yaml
     exported/policy.onnx
     exported/safety_value.onnx
   ```

4. Copy the existing flat `deploy.yaml` and append these terms after `last_action`:

   ```yaml
     ball_position:
       params: {}
       clip: null
       scale: [1.0, 1.0, 1.0]
       history_length: 5

     ball_velocity:
       params: {}
       clip: null
       scale: [1.0, 1.0, 1.0]
       history_length: 5
   ```

5. Switch `config.yaml` to the avoid policy directory and use the avoid safety value model.

The 29-DoF robot action interface does not change. The actor output remains `[1, 29]`; only the actor input changes from `[1, 480]` to `[1, 510]`.

To activate the MoCap avoid path manually, change only the `Velocity` block in the Unitree config:

```yaml
Velocity:
  policy_dir: config/policy/velocity/unitree_avoid_fwd_back_mocap

  mocap_avoid:
    enabled: true
    udp_port: 15151
    max_age_s: 0.10

  safety_value:
    enabled: true
    model_path: config/policy/velocity/unitree_avoid_fwd_back_mocap/exported/safety_value.onnx
```

The bridge must also enable UDP output:

```yaml
udp_enabled: true
udp_host: 127.0.0.1
udp_port: 15151
```

The staged avoid models have been checked:

```text
unitree_avoid_fwd_back_mocap/exported/policy.onnx        [1, 510] -> [1, 29]
unitree_avoid_fwd_back_mocap/exported/safety_value.onnx  [1, 510] -> [1, 1]
unitree_flat_baseline_v4/exported/policy.onnx            [1, 480] -> [1, 29]
```
