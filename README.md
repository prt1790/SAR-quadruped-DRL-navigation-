### `go2_nav/go2_nav_env.py`
The core Gymnasium environment wrapper that connects the RL agents to the Gazebo simulation. Handles:
- LiDAR point cloud processing — slices the 3D scan at ground level into 64 radial beams
- Odometry — reads robot position and orientation from ROS2
- Observation space — 64 LiDAR beams + goal distance + goal angle + sin/cos yaw (68 values total)
- Action space — linear and angular velocity commands in [-1, 1]
- Reward function — progress reward + step penalty + goal bonus + collision penalty
- Episode reset — teleports robot back to spawn position via Gazebo set_pose service

### `go2_nav/champ_gz_bridge.py`
A custom ROS2-to-Gazebo bridge that translates CHAMP quadruped controller joint trajectories into Gazebo native joint position commands. 
Manages all 12 joints of the Go2 (4 legs × 3 joints each: hip, upper leg, lower leg).

### `train_gazebo.py`
The main training script that:
- Initialises the Gazebo environment
- Defines all 6 agents with their hyperparameters
- Runs the training loop with periodic evaluation every 10,000 steps
- Saves model checkpoints
- Generates 5 result plots: learning curves, final return distribution, steps per episode, success/collision rates, and off-policy agent comparison

### `unitree_go2_ros2/disaster_world.sdf`
The Gazebo simulation world containing:
- Collapsed house 3D model (imported from Open Robotics)
- 5 victim location markers (red spheres) at manually surveyed positions inside the building
- 8 rubble boxes and 2 structural pillars as navigation obstacles
- Physics configuration optimised for training speed (4ms step size)

### `unitree_go2_ros2/lidar_4D_lidar.xacro`
Configuration for the Unitree 4D LiDAR L1 sensor, updated to match real hardware specifications (subject to change):
- FOV: 360° horizontal × 90° vertical
- Range: 0.05m–30m
- Update rate: 11Hz
- Noise: Gaussian stddev 0.02m (±2cm accuracy)

### `unitree_go2_ros2/unitree_go2_launch.py`
ROS2 launch file that starts the full simulation stack: Gazebo (headless), robot state publisher, Go2 spawn, ROS2-Gazebo bridge, CHAMP controller, and state estimator.

## Dependencies

- ROS2 Jazzy
- Gazebo Harmonic (gz-sim 8.10.0)
- Python 3.12
- stable-baselines3
- sb3-contrib (for TRPO)
- gymnasium
- torch
- numpy, pandas, matplotlib, seaborn

## How to Run

**Terminal 1 — Launch Gazebo and ROS2 stack:**
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch unitree_go2_sim unitree_go2_launch.py

**Terminal 2 — Start CHAMP bridge:**
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/go2_nav/go2_nav/champ_gz_bridge.py

**Terminal 3 — Start training:**
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/go2_nav
python3 train_gazebo.py


