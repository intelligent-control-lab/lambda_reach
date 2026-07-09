from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("mocap_avoid_bridge"))
    default_config = package_share / "config" / "mocap_avoid_bridge.yaml"

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=str(default_config),
        description="Path to the mocap_avoid_bridge parameter file.",
    )

    bridge_node = Node(
        package="mocap_avoid_bridge",
        executable="mocap_avoid_bridge_node",
        name="mocap_avoid_bridge",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription([config_arg, bridge_node])
