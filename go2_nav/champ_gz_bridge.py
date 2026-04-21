#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from gz.msgs10.double_pb2 import Double
from gz.transport13 import Node as GzNode


# 12 joints of Go2
JOINT_MAP = {
    "lf_hip_joint":       "lf_hip_joint",
    "lf_upper_leg_joint": "lf_upper_leg_joint",
    "lf_lower_leg_joint": "lf_lower_leg_joint",
    "rf_hip_joint":       "rf_hip_joint",
    "rf_upper_leg_joint": "rf_upper_leg_joint",
    "rf_lower_leg_joint": "rf_lower_leg_joint",
    "lh_hip_joint":       "lh_hip_joint",
    "lh_upper_leg_joint": "lh_upper_leg_joint",
    "lh_lower_leg_joint": "lh_lower_leg_joint",
    "rh_hip_joint":       "rh_hip_joint",
    "rh_upper_leg_joint": "rh_upper_leg_joint",
    "rh_lower_leg_joint": "rh_lower_leg_joint",
}

MODEL_NAME = "go2"

class ChampGzBridge(Node):
    def __init__(self):
        super().__init__("champ_gz_bridge")
        self.gz_node = GzNode()
        self.pubs = {}


        # creates a gazebo publisher per joint 
        # each publisher sends position commands to gazebo's joint controller
        for joint in JOINT_MAP.values(): 
            topic = f"/model/{MODEL_NAME}/joint/{joint}/0/cmd_pos"
            pub = self.gz_node.advertise(topic, Double)
            if pub:
                self.pubs[joint] = pub
                self.get_logger().info(f"Advertising {topic}") # bridge start
            else:
                self.get_logger().warn(f"Failed to advertise {topic}")

        self.create_subscription(
            JointTrajectory,
            "/joint_group_effort_controller/joint_trajectory",
            self.traj_callback, 10)

        self.get_logger().info(
            f"Bridge ready - {len(self.pubs)}/12 joints advertised")


    # core translation from CHAMP to Gazebo topic 
    def traj_callback(self, msg): 
        if not msg.points:
            return
        positions = msg.points[0].positions
        msg_out = Double()
        for name, pos in zip(msg.joint_names, positions):
            gz_name = JOINT_MAP.get(name)
            if gz_name and gz_name in self.pubs:
                msg_out.data = float(pos)
                self.pubs[gz_name].publish(msg_out)

def main():
    rclpy.init()
    node = ChampGzBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
