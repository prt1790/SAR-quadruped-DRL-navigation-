import rclpy
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean as GzBoolean
from gz.transport13 import Node as GzNode
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import threading
import math
import time


class Go2NavEnv(gym.Env):
    """
    Gym wrapper around the Go2 Gazebo simulation.

    Observation (matches 2D env layout):
        [0 : num_beams]   normalised LiDAR ring slice  ( / max_range)
        [num_beams]       normalised goal dist          ( / max_goal_dist)
        [num_beams + 1]   normalised goal angle         ( / π)
        [num_beams + 2]   sin(yaw)
        [num_beams + 3]   cos(yaw)

    Action:
        [linear_vel, angular_vel]  both in [-1, 1]
        scaled to max_linear_speed / max_angular_speed before publishing
    """

    def __init__(self,
                 goal_pos=(0.0, 0.0),
                 num_beams=64,
                 max_range=10.0,
                 max_linear_speed=0.5,    # m/s
                 max_angular_speed=1.0,   # rad/s
                 done_threshold=0.3,      # metres
                 collision_threshold=0.4, # metres from lidar hit to count as collision
                 height_band=0.3,         # metres around robot base for slice
                 step_timeout=0.09):      

        super().__init__()

        # 5 fixed victim's positions inside the collapsed house (x, y)
        self._goal_candidates = np.array([
            [-4.61, -5.54],
            [-5.06, -11.97],
            [-2.05, -8.60],
            [-2.17, -1.91],
            [-6.19, -3.13],
        ], dtype=np.float32) # these were measured manually 
        self.goal_pos = self._goal_candidates[0].copy()
        self.num_beams        = num_beams
        self.max_range        = max_range
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed= max_angular_speed
        self.done_threshold   = done_threshold
        self.collision_thresh = collision_threshold
        self.height_band      = height_band
        self.step_timeout     = step_timeout
        self.max_goal_dist    = 10.0  # normalisation constant

        # Gym spaces (identical layout to 2D env)
        # action space also the same (linear and angular velociy)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) 
        obs_dim = num_beams + 4
        # observation space also the same (64 LiDAR beams, goal distance, goal angle, sin & cos )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32) 

        # Internal state
        self._robot_pos   = np.zeros(2)   # x, y
        self._robot_yaw   = 0.0
        self._lidar_ready = threading.Event()
        self._odom_ready  = threading.Event()
        self._latest_ranges = np.full(num_beams, max_range)
        self._prev_dist   = None
        self._lock        = threading.Lock()

        # Gazebo pose reset
        self._gz_node   = GzNode()
        self._spawn_x   = 4.50
        self._spawn_y   = -3.45
        self._spawn_z   = 1.0

        # Start ROS2 in background thread
        rclpy.init()
        self._node = Node('go2_nav_env')
        self._setup_ros()
        self._ros_thread = threading.Thread(
            target=self._spin, daemon=True)
        self._ros_thread.start()

        # Wait for first messages
        print('[Go2NavEnv] Waiting for odom...')
        self._odom_ready.wait(timeout=10.0)
        print('[Go2NavEnv] Waiting for lidar...')
        self._lidar_ready.wait(timeout=10.0)
        print('[Go2NavEnv] Ready.')

  
    def _setup_ros(self):
        self._cmd_pub = self._node.create_publisher(
            Twist, '/cmd_vel', 10)

        self._node.create_subscription(
            Odometry, '/odom',
            self._odom_callback, 10)

        self._node.create_subscription(
            PointCloud2, '/unitree_lidar/points',
            self._lidar_callback, 10)

        # Gazebo reset service (pauses + resets world)
        self._reset_client = self._node.create_client(
            Empty, '/reset_simulation')

    def _spin(self):
        executor = SingleThreadedExecutor()
        executor.add_node(self._node)
        executor.spin()


    def _odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        with self._lock:
            self._robot_pos = np.array([pos.x, pos.y], dtype=np.float32)
            # Convert quaternion → yaw
            siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
            cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
            self._robot_yaw = math.atan2(siny, cosy)
        self._odom_ready.set()

    def _lidar_callback(self, msg):
        # Slices the 3D point cloud at robot height +- height_band, then project surviving points onto num_beams radial bins.
        # How the robot "sees" the obstacles.
        
        # Read all points as a numpy array (x, y, z)
        points = np.array([
            [p[0], p[1], p[2]]
            for p in pc2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
        ], dtype=np.float32)

        ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)

        if len(points) > 0:
            # Distance filter (ignore points closer than 0.5m (robot body))
            dist3d = np.linalg.norm(points, axis=1)
            mask = dist3d > 0.5
            # Height filter (keep points near ground plane)
            mask = mask & (np.abs(points[:, 2]) < self.height_band)
            pts2d = points[mask, :2]   # x, y in lidar frame

            if len(pts2d) > 0:
                angles = np.arctan2(pts2d[:, 1], pts2d[:, 0])  # [-π, π]
                dists  = np.linalg.norm(pts2d, axis=1)

                # Map angle => beam index  (FOV = full 360 degrees)
                beam_idx = ((angles + np.pi) /
                            (2 * np.pi) * self.num_beams).astype(int)
                beam_idx = np.clip(beam_idx, 0, self.num_beams - 1)

                # Minimum distance per beam
                for i, d in zip(beam_idx, dists):
                    if d < ranges[i]:
                        ranges[i] = d

        with self._lock:
            self._latest_ranges = np.clip(ranges, 0.0, self.max_range)
        self._lidar_ready.set()


    def reset(self, seed=None, options=None):
        # stops the robot, telepots it back to the spawn position, and picks a random victim's location
        super().reset(seed=seed)

        # Stop the robot
        self._publish_cmd(0.0, 0.0)
        time.sleep(0.05)

        # Teleport robot back to spawn position using gz set_pose
        self._reset_robot_pose()
        time.sleep(0.15)   # give Gazebo time to settle

        # Randomly select a new victim location each episode
        idx = np.random.randint(len(self._goal_candidates))
        self.goal_pos = self._goal_candidates[idx].copy()
        self._node.get_logger().info(
            f'Episode goal: victim at ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})')



        with self._lock:
            self._prev_dist = float(
                np.linalg.norm(self._robot_pos - self.goal_pos))

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        linear  = float(action[0]) * self.max_linear_speed
        angular = float(action[1]) * self.max_angular_speed
        self._publish_cmd(linear, angular)

        # Wait for next lidar scan
        self._lidar_ready.clear()
        self._lidar_ready.wait(timeout=self.step_timeout)

        obs = self._get_obs()

        with self._lock:
            robot_pos = self._robot_pos.copy()
            ranges    = self._latest_ranges.copy()

        dist_to_goal = float(np.linalg.norm(robot_pos - self.goal_pos))

        # Reward function !! (Same as 2D)
        progress  = (self._prev_dist - dist_to_goal) * 10.0
        reward    = progress - 0.5   # step penalty
        self._prev_dist = dist_to_goal

        # Goal reached
        terminated = dist_to_goal < self.done_threshold
        if terminated:
            reward += 100.0

        # Collision: any beam closer than threshold
        collision = bool(np.any(ranges < self.collision_thresh))
        if collision:
            reward -= 5.0

        return obs, reward, terminated, False, {'collision': collision}

    def close(self):
        self._publish_cmd(0.0, 0.0)
        time.sleep(0.05)
        try:
            self._node.destroy_node()
            rclpy.try_shutdown()
        except Exception:
            pass
    def _publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def _reset_robot_pose(self):
        """Teleport robot back to spawn position via Gazebo set_pose service."""
        req = Pose()
        req.name = 'go2'
        req.position.x = self._spawn_x
        req.position.y = self._spawn_y
        req.position.z = self._spawn_z
        req.orientation.w = 1.0
        req.orientation.x = 0.0
        req.orientation.y = 0.0
        req.orientation.z = 0.0
        result = self._gz_node.request(
            '/world/disaster_world/set_pose', req, Pose,
            GzBoolean, 2000)
        if not result[1]:
            self._node.get_logger().warn('set_pose service call failed')

    def _get_obs(self):
        with self._lock:
            robot_pos = self._robot_pos.copy()
            yaw       = self._robot_yaw
            ranges    = self._latest_ranges.copy()

        # Goal in robot frame
        goal_vec   = self.goal_pos - robot_pos
        goal_dist  = float(np.linalg.norm(goal_vec))
        goal_angle = math.atan2(goal_vec[1], goal_vec[0]) - yaw
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi

        obs = np.concatenate([
            ranges / self.max_range,                  # normalised lidar
            [goal_dist / self.max_goal_dist],         # normalised dist
            [goal_angle / math.pi],                   # normalised angle
            [math.sin(yaw), math.cos(yaw)]            # orientation
        ]).astype(np.float32)

        return obs
