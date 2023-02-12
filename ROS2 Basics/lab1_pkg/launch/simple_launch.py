from launch import LaunchDescription
from launch_ros.actions import Node
import rclpy
import rclpy.node

def generate_launch_description():
    # rclpy.node.Node.declare_parameter('v',5)
    # rclpy.node.Node.declare_parameter('d',10)
    return LaunchDescription([
        Node(
            package='lab1_pkg',
            executable='talker',
            parameters=[{'v': 7.0,'d': 10.0}],
            name='talker'
        ),
        Node(
            package='lab1_pkg',
            executable='relay',
            name='relay'
        )
    ])