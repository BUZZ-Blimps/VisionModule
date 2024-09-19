from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_index',
            default_value='0',
            description='Camera index'
        ),
        DeclareLaunchArgument(
            'calibration_file',
            default_value='',
            description='Path to calibration file (without _elp_left.yaml or _elp_right.yaml)'
        ),
        DeclareLaunchArgument(
            'node_namespace',
            default_value='BurnCreamBlimp',
            description='Node namespace'
        ),
        DeclareLaunchArgument(
            'publish_intermediate',
            default_value='true',
            description='Publish intermediate images'
        ),
        DeclareLaunchArgument(
            'camera_number',
            default_value='1',
            description='Camera number for calibration file naming'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='yolov5.rknn',
            description='Path to rknn model file'
        ),
        Node(
            package='stereo_vision',
            executable='stereo_vision',
            name='stereo_vision',
            parameters=[{
                'camera_index': LaunchConfiguration('camera_index'),
                'calibration_file': LaunchConfiguration('calibration_file'),
                'node_namespace': LaunchConfiguration('node_namespace'),
                'publish_intermediate': LaunchConfiguration('publish_intermediate'),
                'camera_number': LaunchConfiguration('camera_number'),
                'model_path': LaunchConfiguration('model_path'),
            }],
            output='screen'
        )
    ])