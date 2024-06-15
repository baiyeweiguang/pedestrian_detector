import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('pedestrian_detector'),
        'config',
        'pedestrian_detector.yaml'
    )

    return LaunchDescription([
        Node(
            package='pedestrian_detector',
            executable='pedestrian_detector_node',
            name='pedestrian_detector',
            output='screen',
            parameters=[config]
        )
    ])

