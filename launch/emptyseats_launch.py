from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
        # Path to the RealSense launch file
    realsense_launch_file_dir = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )
    robot_localization_launch_file_dir = os.path.join(get_package_share_directory('robot_localization'), 'launch')

    return LaunchDescription([
        DeclareLaunchArgument('offline', default_value='false'),

        # Static transform for the first camera
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_camera1',
            arguments=['0', '0', '0', '0', '0', '-1.57', 'base_link', 'camera_link']
        ),
        

        # Include the RealSense camera launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file_dir),
            launch_arguments={
                'enable_sync': 'true',
                'align_depth.enable': 'true',
                'enable_color': 'true',
                'enable_gyro': 'true',
                'enable_accel': 'true',
                'unite_imu_method': '2',
                'pointcloud.enable': 'true',
                'pointcloud.allow_no_texture_points': 'true',
                'pointcloud.stream_filter': '0',
                'rgb_camera.color_profile': '640,480,30',
                'depth_module.depth_profile': '640,480,30',
                'temporal_filter.enable': 'true',
                'hole_filling_filter.enable': 'true'
            }.items(),
        ),

        # Imu Filter Node
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='ImuFilter',
            parameters=[{
                'use_mag': False,
                'publish_tf': False,
                'world_frame': 'enu',
                'enable_gyro': 'true',
                'gyro_fps': '200',  # Increase the frequency
                'enable_accel': 'true',
                'accel_fps': '200',  # Increase the frequency
            }],
            remappings=[
                ('/imu/data_raw', '/camera/camera/imu')
            ]
        ),

        # Launch RTAB-Map Odometry
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rtabmap_odometry',
            output='screen',
            remappings=[
                ('rgb/image', '/camera/camera/color/image_raw'),
                ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
                ('rgb/camera_info', '/camera/camera/color/camera_info'),
                ('imu', '/camera/camera/imu'),
            ],
            parameters=[
                {'frame_id': 'base_link'},
                {'wait_imu_to_init': True}
            ]
        ),

        # Include Robot Localization UKF Template Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([robot_localization_launch_file_dir, '/ukf.launch.py']),
        ),

        # Robot Localization Parameters
        Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_se',
            parameters=[{
                'frequency': 300.0,
                'base_link_frame': 'base_link',
                'odom0': '/odom',
                'odom0_config': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                'odom0_relative': True,
                'odom0_pose_rejection_threshold': 10000000.0,
                'odom0_twist_rejection_threshold': 10000000.0,
                'imu0': '/imu/data',
                'imu0_config': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                'imu0_differential': True,
                'imu0_relative': False,
                'use_control': False,
            }]
        ),
        # Navigation Node
        Node(
            package='buttonlistener',
            executable='listener',
            name='button_listener',
            output='screen'
        ),
        # empty seats
        Node(
            package='emptyseats',
            executable='emptyseats',
            name='empty_seats',
            output='screen'
        ),
        # Navigation Node
        Node(
            package='btn',
            executable='emptyseats',
            name='btn_emptyseats',
            output='screen'
        ),
        
    ])
