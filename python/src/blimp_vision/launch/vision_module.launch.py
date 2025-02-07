from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

blimp_name = 'SillyAh'

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_number',
            default_value='12',
            description='Camera device number'
        ),
        DeclareLaunchArgument(
            'device_path',
            default_value='/dev/video0',
            description='Camera device path'
        ),
        DeclareLaunchArgument(
            'calibration_path',
            default_value=PathJoinSubstitution([
                get_package_share_directory('blimp_vision'),
                'calibration'
            ]),
            description='Path to camera calibration file'
        ),
        DeclareLaunchArgument(
            'ball_model_file',
            default_value=PathJoinSubstitution([
                get_package_share_directory('blimp_vision'),
                'models/balloon/balloon_v11n_rknn_model'
            ]),
            description='Path to ball detection model'
        ),
        DeclareLaunchArgument(
            'goal_model_file',
            default_value=PathJoinSubstitution([
                get_package_share_directory('blimp_vision'),
                'models/goal/goal_v11n_rknn_model'
            ]),
            description='Path to goal detection model'
        ),
        DeclareLaunchArgument(
            'verbose_mode',
            default_value='false',
            description='Enable verbose logging'
        ),
        DeclareLaunchArgument(
            'save_frames',
            default_value='false',
            description='Enable frame saving'
        ),
        DeclareLaunchArgument(
            'save_location',
            default_value='frames/',
            description='Location to save frames'
        ),
        Node(
            package='blimp_vision',
            executable='blimp_vision_node',
            name='blimp_vision_node',
            namespace=blimp_name,
            parameters=[{
                'camera_number': LaunchConfiguration('camera_number'),
                'device_path': LaunchConfiguration('device_path'),
                'calibration_path': LaunchConfiguration('calibration_path'),
                'ball_model_file': LaunchConfiguration('ball_model_file'),
                'goal_model_file': LaunchConfiguration('goal_model_file'),
                'verbose_mode': LaunchConfiguration('verbose_mode'),
                'save_frames': LaunchConfiguration('save_frames'),
                'save_location': LaunchConfiguration('save_location')
            }],
            output='screen'
        )
    ])
