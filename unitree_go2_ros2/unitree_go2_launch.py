import os
import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    base_frame = "base_link"

    unitree_go2_sim = launch_ros.substitutions.FindPackageShare(
        package="unitree_go2_sim").find("unitree_go2_sim")
    unitree_go2_description = launch_ros.substitutions.FindPackageShare(
        package="unitree_go2_description").find("unitree_go2_description")

    joints_config    = os.path.join(unitree_go2_sim, "config/joints/joints.yaml")
    gait_config      = os.path.join(unitree_go2_sim, "config/gait/gait.yaml")
    links_config     = os.path.join(unitree_go2_sim, "config/links/links.yaml")
    ros_control_config = os.path.join(unitree_go2_sim, "config/ros_control/ros_control.yaml")
    default_model_path = os.path.join(unitree_go2_description, "urdf/unitree_go2_robot.xacro")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="true")
    declare_world_init_x = DeclareLaunchArgument("world_init_x", default_value="4.50")
    declare_world_init_y = DeclareLaunchArgument("world_init_y", default_value="-3.45")
    declare_world_init_z = DeclareLaunchArgument("world_init_z", default_value="1.0")
    declare_world_init_heading = DeclareLaunchArgument("world_init_heading", default_value="0.0")
    declare_robot_name = DeclareLaunchArgument("robot_name", default_value="go2")

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", default_model_path,
                     " robot_controllers:=", ros_control_config]),
            value_type=str)
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    quadruped_controller_node = Node(
        package="champ_base",
        executable="quadruped_controller_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"gazebo": True},
            {"publish_joint_states": True},
            {"publish_joint_control": True},
            {"publish_foot_contacts": False},
            {"joint_controller_topic": "joint_group_effort_controller/joint_trajectory"},
            {"urdf": ParameterValue(Command(["xacro ", default_model_path]), value_type=str)},
            joints_config, links_config, gait_config,
            {"hardware_connected": False},
            {"close_loop_odom": True},
        ],
        remappings=[("/cmd_vel/smooth", "/cmd_vel")],
    )

    state_estimator_node = Node(
        package="champ_base",
        executable="state_estimation_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"orientation_from_imu": True},
            {"urdf": ParameterValue(Command(["xacro ", default_model_path]), value_type=str)},
            joints_config, links_config, gait_config,
        ],
    )

    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x','0','--y','0','--z','0',
                   '--roll','0','--pitch','0','--yaw','0',
                   '--frame-id','map','--child-frame-id','odom'],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    base_footprint_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x','0','--y','0','--z','0',
                   '--roll','0','--pitch','0','--yaw','0',
                   '--frame-id','base_footprint','--child-frame-id','base_link'],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': [PathJoinSubstitution([
                unitree_go2_description, 'worlds', 'disaster_world.sdf'
            ]), ' -s']
        }.items(),
    )

    gazebo_spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', LaunchConfiguration('robot_name'),
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('world_init_x'),
            '-y', LaunchConfiguration('world_init_y'),
            '-z', LaunchConfiguration('world_init_z'),
            '-Y', LaunchConfiguration('world_init_heading'),
        ],
    )

    gazebo_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/unitree_lidar/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/joint_group_effort_controller/joint_trajectory@trajectory_msgs/msg/JointTrajectory]gz.msgs.JointTrajectory',
        ],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_world_init_x,
        declare_world_init_y,
        declare_world_init_z,
        declare_world_init_heading,
        declare_robot_name,
        gz_sim,
        robot_state_publisher_node,
        gazebo_spawn_robot,
        gazebo_bridge,
        quadruped_controller_node,
        state_estimator_node,
        map_to_odom_tf,
        base_footprint_tf,
    ])
