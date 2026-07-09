from __future__ import annotations

import math
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # ROS order: x, y, z, w


@dataclass(frozen=True)
class PoseSample:
    stamp_s: float
    received_s: float
    pos: Vector3
    quat_xyzw: Quaternion


@dataclass(frozen=True)
class RigidBodyState:
    pose: PoseSample
    vel_w: Vector3


@dataclass(frozen=True)
class AvoidObservation:
    stamp_s: float
    valid: bool
    ball_pos_base: Vector3
    ball_vel_base: Vector3
    reason: str = ""


class VelocityEstimator:
    """Finite-difference velocity estimator with EMA smoothing."""

    def __init__(self, alpha: float, min_dt_s: float, max_dt_s: float, max_velocity_mps: float):
        self.alpha = max(0.0, min(1.0, alpha))
        self.min_dt_s = min_dt_s
        self.max_dt_s = max_dt_s
        self.max_velocity_mps = max_velocity_mps
        self._prev_pos: Optional[Vector3] = None
        self._prev_stamp_s: Optional[float] = None
        self._filtered_vel: Vector3 = (0.0, 0.0, 0.0)

    def update(self, pos: Vector3, stamp_s: float) -> Vector3:
        if self._prev_pos is None or self._prev_stamp_s is None:
            self._prev_pos = pos
            self._prev_stamp_s = stamp_s
            return self._filtered_vel

        dt = stamp_s - self._prev_stamp_s
        if dt < self.min_dt_s or dt > self.max_dt_s:
            self._prev_pos = pos
            self._prev_stamp_s = stamp_s
            return self._filtered_vel

        raw = tuple((pos[i] - self._prev_pos[i]) / dt for i in range(3))
        raw = clip_norm(raw, self.max_velocity_mps)
        self._filtered_vel = tuple(
            self.alpha * raw[i] + (1.0 - self.alpha) * self._filtered_vel[i] for i in range(3)
        )
        self._prev_pos = pos
        self._prev_stamp_s = stamp_s
        return self._filtered_vel


class UdpAvoidObservationPublisher:
    """Optional binary publisher for a non-ROS controller process.

    Packet format, little endian:
      magic[4] = b"MAVO"
      version:uint16 = 1
      payload_size:uint16 = sizeof(packet)
      seq:uint64
      stamp_s:double
      valid:uint8
      padding[3]
      ball_pos_base:float32[3]
      ball_vel_base:float32[3]
    """

    _PACKET = struct.Struct("<4sHHQdB3x6f")
    _MAGIC = b"MAVO"
    _VERSION = 1

    def __init__(self, host: str, port: int):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, obs: AvoidObservation) -> None:
        self._seq += 1
        values = (*obs.ball_pos_base, *obs.ball_vel_base)
        packet = self._PACKET.pack(
            self._MAGIC,
            self._VERSION,
            self._PACKET.size,
            self._seq,
            obs.stamp_s,
            1 if obs.valid else 0,
            *values,
        )
        self._sock.sendto(packet, self._addr)


class MocapAvoidBridge(Node):
    def __init__(self) -> None:
        super().__init__("mocap_avoid_bridge")

        self._base_pose_topic = self.declare_parameter("base_pose_topic", "/g1_base/pose").value
        self._ball_pose_topic = self.declare_parameter("ball_pose_topic", "/avoid_ball/pose").value
        self._output_topic = self.declare_parameter("output_topic", "/avoid/mocap_observation").value
        self._valid_topic = self.declare_parameter("valid_topic", "/avoid/mocap_valid").value
        self._publish_debug_twist = bool(self.declare_parameter("publish_debug_twist", True).value)
        self._base_twist_topic = self.declare_parameter("base_twist_topic", "/avoid/debug/base_twist").value
        self._ball_twist_topic = self.declare_parameter("ball_twist_topic", "/avoid/debug/ball_twist").value

        output_rate_hz = float(self.declare_parameter("output_rate_hz", 50.0).value)
        self._max_pose_age_s = float(self.declare_parameter("max_pose_age_s", 0.10).value)
        self._max_stamp_delta_s = float(self.declare_parameter("max_stamp_delta_s", 0.04).value)
        velocity_filter_alpha = float(self.declare_parameter("velocity_filter_alpha", 0.35).value)
        velocity_min_dt_s = float(self.declare_parameter("velocity_min_dt_s", 0.001).value)
        velocity_max_dt_s = float(self.declare_parameter("velocity_max_dt_s", 0.20).value)
        max_velocity_mps = float(self.declare_parameter("max_velocity_mps", 20.0).value)

        udp_enabled = bool(self.declare_parameter("udp_enabled", False).value)
        udp_host = self.declare_parameter("udp_host", "127.0.0.1").value
        udp_port = int(self.declare_parameter("udp_port", 15151).value)

        self._lock = threading.Lock()
        self._base_state: Optional[RigidBodyState] = None
        self._ball_state: Optional[RigidBodyState] = None
        self._base_velocity = VelocityEstimator(
            velocity_filter_alpha, velocity_min_dt_s, velocity_max_dt_s, max_velocity_mps
        )
        self._ball_velocity = VelocityEstimator(
            velocity_filter_alpha, velocity_min_dt_s, velocity_max_dt_s, max_velocity_mps
        )
        self._last_valid_observation = AvoidObservation(
            stamp_s=0.0,
            valid=False,
            ball_pos_base=(0.0, 0.0, 0.0),
            ball_vel_base=(0.0, 0.0, 0.0),
            reason="not initialized",
        )

        self._obs_pub = self.create_publisher(Float32MultiArray, self._output_topic, 10)
        self._valid_pub = self.create_publisher(Bool, self._valid_topic, 10)
        self._base_twist_pub = (
            self.create_publisher(TwistStamped, self._base_twist_topic, 10) if self._publish_debug_twist else None
        )
        self._ball_twist_pub = (
            self.create_publisher(TwistStamped, self._ball_twist_topic, 10) if self._publish_debug_twist else None
        )
        self._udp_pub = UdpAvoidObservationPublisher(udp_host, udp_port) if udp_enabled else None

        self.create_subscription(PoseStamped, self._base_pose_topic, self._on_base_pose, 20)
        self.create_subscription(PoseStamped, self._ball_pose_topic, self._on_ball_pose, 20)

        timer_period = 1.0 / max(output_rate_hz, 1.0)
        self.create_timer(timer_period, self._publish_latest)

        self.get_logger().info(
            "MoCap avoid bridge started: "
            f"base={self._base_pose_topic}, ball={self._ball_pose_topic}, "
            f"output={self._output_topic}, udp={'on' if udp_enabled else 'off'}"
        )

    def _on_base_pose(self, msg: PoseStamped) -> None:
        sample = pose_sample_from_msg(msg, self.get_clock().now().nanoseconds * 1.0e-9)
        with self._lock:
            vel = self._base_velocity.update(sample.pos, sample.stamp_s)
            self._base_state = RigidBodyState(sample, vel)

    def _on_ball_pose(self, msg: PoseStamped) -> None:
        sample = pose_sample_from_msg(msg, self.get_clock().now().nanoseconds * 1.0e-9)
        with self._lock:
            vel = self._ball_velocity.update(sample.pos, sample.stamp_s)
            self._ball_state = RigidBodyState(sample, vel)

    def _publish_latest(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1.0e-9
        obs = self._compute_observation(now_s)

        valid_msg = Bool()
        valid_msg.data = obs.valid
        self._valid_pub.publish(valid_msg)

        if obs.valid:
            self._obs_pub.publish(observation_msg(obs))
            self._last_valid_observation = obs
        elif self._udp_pub is None:
            self.get_logger().warn(
                f"Invalid MoCap avoid observation: {obs.reason}",
                throttle_duration_sec=1.0,
            )

        if self._udp_pub is not None:
            try:
                udp_obs = obs if obs.valid else AvoidObservation(
                    stamp_s=now_s,
                    valid=False,
                    ball_pos_base=self._last_valid_observation.ball_pos_base,
                    ball_vel_base=self._last_valid_observation.ball_vel_base,
                    reason=obs.reason,
                )
                self._udp_pub.send(udp_obs)
            except OSError as exc:
                self.get_logger().warn(f"Failed to send MoCap UDP packet: {exc}", throttle_duration_sec=1.0)

        if self._publish_debug_twist:
            self._publish_debug_twists()

    def _compute_observation(self, now_s: float) -> AvoidObservation:
        with self._lock:
            base = self._base_state
            ball = self._ball_state

        if base is None or ball is None:
            return AvoidObservation(now_s, False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "missing rigid body pose")

        base_age = now_s - base.pose.received_s
        ball_age = now_s - ball.pose.received_s
        if base_age > self._max_pose_age_s or ball_age > self._max_pose_age_s:
            return AvoidObservation(
                now_s,
                False,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                f"stale pose base_age={base_age:.3f}s ball_age={ball_age:.3f}s",
            )

        stamp_delta = abs(base.pose.stamp_s - ball.pose.stamp_s)
        if stamp_delta > self._max_stamp_delta_s:
            return AvoidObservation(
                now_s,
                False,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                f"unsynchronized poses stamp_delta={stamp_delta:.3f}s",
            )

        rel_pos_w = sub3(ball.pose.pos, base.pose.pos)
        rel_vel_w = sub3(ball.vel_w, base.vel_w)
        ball_pos_base = quat_rotate_inverse(base.pose.quat_xyzw, rel_pos_w)
        ball_vel_base = quat_rotate_inverse(base.pose.quat_xyzw, rel_vel_w)
        return AvoidObservation(
            stamp_s=max(base.pose.stamp_s, ball.pose.stamp_s),
            valid=True,
            ball_pos_base=ball_pos_base,
            ball_vel_base=ball_vel_base,
        )

    def _publish_debug_twists(self) -> None:
        with self._lock:
            base = self._base_state
            ball = self._ball_state

        if base is not None and self._base_twist_pub is not None:
            self._base_twist_pub.publish(twist_msg(base.pose, base.vel_w, "g1_base_mocap"))
        if ball is not None and self._ball_twist_pub is not None:
            self._ball_twist_pub.publish(twist_msg(ball.pose, ball.vel_w, "avoid_ball_mocap"))


def pose_sample_from_msg(msg: PoseStamped, received_s: float) -> PoseSample:
    stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
    if stamp_s <= 0.0:
        stamp_s = received_s

    return PoseSample(
        stamp_s=stamp_s,
        received_s=received_s,
        pos=(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
        quat_xyzw=normalize_quat(
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            )
        ),
    )


def observation_msg(obs: AvoidObservation) -> Float32MultiArray:
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="ball_position_base_then_velocity_base", size=6, stride=6),
    ]
    msg.data = [float(v) for v in (*obs.ball_pos_base, *obs.ball_vel_base)]
    return msg


def twist_msg(pose: PoseSample, vel: Vector3, frame_id: str) -> TwistStamped:
    msg = TwistStamped()
    sec = int(math.floor(pose.stamp_s))
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((pose.stamp_s - sec) * 1.0e9)
    msg.header.frame_id = frame_id
    msg.twist.linear.x = vel[0]
    msg.twist.linear.y = vel[1]
    msg.twist.linear.z = vel[2]
    return msg


def sub3(lhs: Vector3, rhs: Vector3) -> Vector3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def clip_norm(v: Vector3, max_norm: float) -> Vector3:
    if max_norm <= 0.0:
        return v
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm <= max_norm or norm == 0.0:
        return v
    scale = max_norm / norm
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def normalize_quat(q: Quaternion) -> Quaternion:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_rotate_inverse(q_xyzw: Quaternion, v: Vector3) -> Vector3:
    """Rotate a world-frame vector into the body frame represented by q_xyzw."""
    x, y, z, w = normalize_quat(q_xyzw)
    # Equivalent to q.conjugate() * [v, 0] * q.
    q_inv = (-x, -y, -z, w)
    return quat_rotate(q_inv, v)


def quat_rotate(q_xyzw: Quaternion, v: Vector3) -> Vector3:
    x, y, z, w = normalize_quat(q_xyzw)
    vx, vy, vz = v

    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # v' = v + w * t + cross(q_xyz, t)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = MocapAvoidBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
