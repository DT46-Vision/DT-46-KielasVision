from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('rm_detector')
    cls_model = os.path.join(pkg_share, 'model', 'mlp.onnx')

    return LaunchDescription([
        Node(
            package='rm_detector',
            executable='rm_detector_node',
            name='rm_detector',
            output='screen',
            parameters=[{
                'cls_model_path': cls_model,
                'cls_invert_binary': False,
            }]
        )
    ])
