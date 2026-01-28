#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from mavros_msgs.msg import State, PositionTarget
from std_msgs.msg import Empty, Float32MultiArray
import ros2_numpy as rnp
import time
import cv2
import numpy as np
import torch
import struct
from scipy.spatial.transform import Rotation as R

# Sample Factory inference (your standalone file)
from standalone_inference import SF_model_initializer

from scipy.ndimage import median_filter


def ssa(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ============================================================================
# Configuration matching lidar_navigation_task training
# ============================================================================
class Config:
    # Image settings (match training)
    IMAGE_HEIGHT = 48  # After (3,6) max pooling from 135
    IMAGE_WIDTH = 80   # After (3,6) max pooling from 480
    IMAGE_MAX_DEPTH = 10.0
    IMAGE_MIN_DEPTH = 0.02
    
    # Observation dimensions
    STATE_DIM = 17
    LIDAR_DIM = 16 * 20  # 320 (downsampled lidar grid)
    TOTAL_OBS_DIM = STATE_DIM + LIDAR_DIM  # 337
    
    # Action dimensions
    ACTION_DIM = 4
    
    # ROS topics
    IMAGE_TOPIC = "/m100/front/depth_image"
    POINTCLOUD_TOPIC = "/rslidar_points"
    ODOM_TOPIC = "/msf_core/odometry"
    ACTION_TOPIC = "/cmd_vel"
    TARGET_TOPIC = "/target"
    MAVROS_STATE_TOPIC = "/rmf/mavros/state"
    PATH_TOPIC = "/gbplanner_path"
    MAVROS_CMD_TOPIC = "/mavros/setpoint_raw/local"
    
    # Action transformation (match your training config)
    # These should match the action_transformation_function in your task config
    ACTION_SCALE = np.array([1.0, 1.0, 0.5, 1.0])  # m/s 
    # ACTION_SCALE = np.array([0.5, 0.5, 0.25, 0.75])  # m/s
    
    # Frame IDs
    BODY_FRAME_ID = "mimosa_body"
    
    # Control
    USE_MAVROS_STATE = False
    ACTION_FILTER_ALPHA = np.array([0.3, 0.3, 0.65, 0.3])  # EMA filter
    
    # Device
    DEVICE = "cuda:0"  # Default device, can be overridden by command line arg

    # Lidar
    LIDAR_MAX_RANGE = 10.0
    LIDAR_MIN_RANGE = 0.4

    MEDIAN_FILTER = True
    MEDIAN_FILTER_KERNEL_SIZE = 7

    # Reset policy at new waypoint
    RESET_AT_NEW_WP = False

cfg = Config()

# ============================================================================
# EMA Filter (same as before)
# ============================================================================
class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None
    
    def reset(self):
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

# ============================================================================
# Filtering of Bins for Removing point Noise
# ============================================================================

def bin_filter(bins, kernel_size=3):
    if cfg.MEDIAN_FILTER == True:
        bins_filtered = median_filter(bins.cpu().numpy(), size=kernel_size)
        return torch.from_numpy(bins_filtered).to(bins.device)
    else:
        return bins

# ============================================================================
# LiDAR Binning and Downsampling from PointCloud2
# ============================================================================

def binning_and_downsampling(points3d_np):
    # Convert to spherical coordinates
    x = points3d_np[:, 0]
    y = points3d_np[:, 1]
    z = points3d_np[:, 2]
    x[np.isnan(x)] = 100.0
    y[np.isnan(y)] = 100.0
    z[np.isnan(z)] = 100.0
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x) % (2 * np.pi)  # azimuth
    phi = np.arccos(z / (r + 1e-6))  # elevation
    
    r[r < cfg.LIDAR_MIN_RANGE] = cfg.LIDAR_MAX_RANGE
    r[r > cfg.LIDAR_MAX_RANGE] = cfg.LIDAR_MAX_RANGE
    r[np.isnan(r)] = cfg.LIDAR_MAX_RANGE
    r[np.isinf(r)] = cfg.LIDAR_MAX_RANGE
    spherical_points = torch.from_numpy(np.stack([r, theta, phi], axis=1)).to(cfg.DEVICE)  # (N, 3)
    azimuth_bins = 480
    elevation_bins = 48
    # azimuth goes from 0 to 2pi
    azimuth_idx = (spherical_points[:, 1] / (2 * torch.pi) * azimuth_bins)
    # elevation goes from 0 to pi
    elevation_idx = (spherical_points[:, 2] / (torch.pi/2) * elevation_bins)


    # Clamp indices to avoid out-of-bounds
    azimuth_idx = torch.clamp(azimuth_idx, 0, azimuth_bins - 1).long()
    elevation_idx = torch.clamp(elevation_idx, 0, elevation_bins - 1).long()

    azimuth_idx[azimuth_idx < 0] = 0
    elevation_idx[elevation_idx < 0] = 0

    # Initialize bins with infinity
    bins = torch.full((azimuth_bins, elevation_bins), float('inf'))

    # Flatten the indices for scatter_reduce
    flat_indices = (azimuth_idx * elevation_bins + elevation_idx)
    input_tensor = torch.full((azimuth_bins * elevation_bins,), 50.0).to(cfg.DEVICE)
    try:
        # Use scatter_reduce to compute the minimum r per bin
        bins = torch.scatter_reduce(
            input=input_tensor,
            dim=0,
            index=flat_indices,
            src=spherical_points[:, 0],
            reduce='amin',
            include_self=False
        )
        # try filtering the bins to remove noise
        bins = bins.view(azimuth_bins, elevation_bins)
        bins_filter = bin_filter(bins, kernel_size=cfg.MEDIAN_FILTER_KERNEL_SIZE)
        bins2 = bins_filter.T.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        bins2_flipped = torch.flip(bins2, [3])  # flip horizontally to match original orientation
        bins_downsampled = -torch.nn.functional.max_pool2d(
            -bins2_flipped,
            kernel_size=(bins2.shape[2]//16, bins2.shape[3]//20)
        )
    except Exception as e:
        print("Error during scatter_reduce:", e)
        bins_downsampled = torch.full((1, 1, 16, 20), cfg.LIDAR_MAX_RANGE)
    return bins, bins_downsampled

   

# ============================================================================
# Parse Pointcloud Optimized
# ============================================================================

def parse_pointcloud_optimized(msg):
    """
    Most optimized version - checks format first, then uses best method
    This is the RECOMMENDED method
    """
    pc = rnp.numpify(msg) #pc.shape=(720,1280)
    height = pc.shape[0]
    width = pc.shape[1]
    points3d_np = np.zeros((height * width, 3), dtype=np.float32)
    points3d_np[:, 0] = np.resize(pc['x'], height * width)
    points3d_np[:, 1] = np.resize(pc['y'], height * width)
    points3d_np[:, 2] = np.resize(pc['z'], height * width)
    return points3d_np


# ============================================================================
# Main ROS Node
# ============================================================================
class LidarNavigationNode(Node):
    def __init__(self, device="cuda:0"):
        super().__init__('lidar_navigation_node')
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load Sample Factory model on specified device
        self.policy = SF_model_initializer()

        # State variables
        self.position = np.zeros(3)
        self.target_position = np.zeros(3)
        self.target_position[0] = 5.0
        self.target_position[2] = 2.0
        self.target_yaw = 0.0
        self.rpy = np.zeros(3)
        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.prev_action = np.zeros(cfg.ACTION_DIM)
        self.downsampled_lidar = np.zeros(cfg.LIDAR_DIM)

        self.state_obs_cpu = torch.zeros(cfg.STATE_DIM, device="cpu", dtype=torch.float32)
        self.obs_gpu = torch.zeros(cfg.TOTAL_OBS_DIM, device=self.device, dtype=torch.float32)

        self.obs_cpu = torch.zeros(cfg.TOTAL_OBS_DIM, device="cpu", dtype=torch.float32)
        self.obs = torch.zeros(cfg.TOTAL_OBS_DIM, device=self.device, dtype=torch.float32)

        self.lidar_tensor = torch.zeros(cfg.LIDAR_DIM, device=self.device, dtype=torch.float32)
        
        # Control state
        self.enable = False
        self.action_filter = EMA(alpha=cfg.ACTION_FILTER_ALPHA)
        
        # QoS profile for reliable delivery
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.action_pub = self.create_publisher(Twist, cfg.ACTION_TOPIC, reliable_qos)
        self.action_viz_pub = self.create_publisher(TwistStamped, cfg.ACTION_TOPIC + "_viz", 1)
        self.filtered_action_pub = self.create_publisher(Twist, cfg.ACTION_TOPIC + "_filtered", reliable_qos)
        self.local_setpoint_pub = self.create_publisher(PositionTarget, cfg.MAVROS_CMD_TOPIC, reliable_qos)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, cfg.IMAGE_TOPIC, self.image_callback, 1)
        # self.pointcloud_sub = self.create_subscription(PointCloud2, cfg.POINTCLOUD_TOPIC, self.pointcloud_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, cfg.ODOM_TOPIC, self.odom_callback, 1)
        self.target_sub = self.create_subscription(PoseStamped, cfg.TARGET_TOPIC, self.target_callback, 1)
        self.path_sub = self.create_subscription(Path, cfg.PATH_TOPIC, self.path_callback, 1)
        self.reset_sub = self.create_subscription(Empty, "/reset", self.reset_callback, 1)
        
        if cfg.USE_MAVROS_STATE:
            self.state_sub = self.create_subscription(State, cfg.MAVROS_STATE_TOPIC, self.state_callback, 1)
        
        # New subscription for pre-processed lidar
        self.processed_lidar_sub = self.create_subscription(
            Float32MultiArray, "/processed_lidar", self.processed_lidar_callback, 1)
        
        self.get_logger().info("Lidar Navigation Node initialized")
    
    def odom_callback(self, msg):
        """Extract odometry data"""
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Extract orientation (quaternion -> euler)
        q = msg.pose.pose.orientation
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        self.rpy = ssa(rot.as_euler('xyz', degrees=False))
        
        # Body frame velocities
        self.body_lin_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
        self.body_ang_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])
    
    def reset_policy(self):
        # Reset on new target
        self.policy.reset()
        self.prev_action = np.zeros(cfg.ACTION_DIM)
        self.action_filter.reset()

    
    def target_callback(self, msg):
        """Update target position"""
        self.target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        # Extract target yaw
        q = msg.pose.orientation
        rot = R.from_quat([q.x, q.y, q.z, q.w], scalar_first=False)
        self.target_yaw = ssa(rot.as_euler('xyz', degrees=False)[2])

        # Reset on new target
        if cfg.RESET_AT_NEW_WP:
            self.reset_policy()
        self.get_logger().info(f"New target: {self.target_position}, yaw: {self.target_yaw:.3f}")
    
    def path_callback(self, msg):
        """Extract first pose from path and update target"""
        if len(msg.poses) > 0:
            self.target_callback(msg.poses[-1])
    
    def reset_callback(self, msg):
        """Reset network state"""
        self.reset_policy()
        self.get_logger().info("Network reset")
    
    def state_callback(self, msg):
        """Check MAVROS state"""
        if cfg.USE_MAVROS_STATE:
            was_enabled = self.enable
            self.enable = (msg.mode == "OFFBOARD" or msg.mode == "GUIDED")
            if self.enable and not was_enabled:
                self.reset_callback(None)
    
    def prepare_observation(self):
        """
        Prepare observation matching lidar_navigation_task:
        - [0:3]: unit vector to target (in vehicle frame)
        - [3]: distance to target
        - [4]: roll
        - [5]: pitch
        - [6]: 0.0 (placeholder)
        - [7:10]: body linear velocity
        - [10:13]: body angular velocity
        - [13:17]: previous actions
        - [17:337]: downsampled lidar (320 dims)
        
        Optimized: Fill state obs on CPU, transfer ONCE to GPU along with lidar
        """
        # Compute vector to target in vehicle frame (yaw-only rotation)
        vec_to_target = self.target_position - self.position
        vehicle_yaw = self.rpy[2]
        vehicle_rot = R.from_euler('z', vehicle_yaw)
        vec_to_target_vehicle = vehicle_rot.inv().apply(vec_to_target)
        
        # Distance to target
        dist_to_target = np.linalg.norm(vec_to_target_vehicle)
        # clamped_dist = np.clip(dist_to_target, 0.0, 3.0)
        clamped_dist = np.clip(dist_to_target, 0.0, 7.0)
        
        # Unit vector to target
        unit_vec_to_target = vec_to_target_vehicle / (dist_to_target + 1e-6)
        
        # Fill state observation on CPU first (17 dims)
        self.state_obs_cpu[0:3] = torch.from_numpy(unit_vec_to_target).float()
        self.state_obs_cpu[3] = clamped_dist
        self.state_obs_cpu[4] = self.rpy[0]  # roll
        self.state_obs_cpu[5] = self.rpy[1]  # pitch
        self.state_obs_cpu[6] = ssa(self.target_yaw - self.rpy[2])  # yaw to target
        self.state_obs_cpu[7:10] = torch.from_numpy(self.body_lin_vel).float()
        self.state_obs_cpu[10:13] = torch.from_numpy(self.body_ang_vel).float()
        self.state_obs_cpu[13:17] = torch.from_numpy(self.prev_action).float()
        
        # Transfer state obs to GPU and fill into obs_gpu (SINGLE TRANSFER)
        self.obs_gpu[0:cfg.STATE_DIM] = self.state_obs_cpu.to(self.device)
        
        # Fill lidar (already on GPU, no transfer!)
        self.obs_gpu[cfg.STATE_DIM:] = self.lidar_tensor
        
        return self.obs_gpu
        
        return np.array([vel_x, vel_y, vel_z, yaw_rate])
    
    def transform_action(self, action):
        """
        Transform network output to velocity commands
        Match your training's action_transformation_function
        """
        # Assuming action is in [-1, 1] range
        # Transform to body frame velocities
        # clamp action first
        action = np.clip(action, -1.0, 1.0)
        action[0:3] = 2.0*(action[0:3])
        action[3] = np.pi/3.0*(action[3])
        scaled_action = action * cfg.ACTION_SCALE
        return scaled_action

    
    def publish_action(self, action):
        """Publish action as Twist message"""
        # Transform action
        vel_cmd = self.transform_action(action)
        self.prev_action = vel_cmd.copy()
        # print("Publishing action:", vel_cmd)
        
        # Apply EMA filter
        filtered_vel = self.action_filter.update(vel_cmd)
        
        # Create Twist message
        twist_msg = Twist()
        twist_msg.linear.x = filtered_vel[0]
        twist_msg.linear.y = filtered_vel[1]
        twist_msg.linear.z = filtered_vel[2]
        twist_msg.angular.z = filtered_vel[3]

        # print publising action:
        print("Publishing action:", twist_msg)
        
        # Publish
        self.filtered_action_pub.publish(twist_msg)
        self.action_pub.publish(twist_msg)
        
        # Publish visualization
        viz_msg = TwistStamped()
        viz_msg.header.stamp = self.get_clock().now().to_msg()
        viz_msg.header.frame_id = cfg.BODY_FRAME_ID
        viz_msg.twist = twist_msg
        self.action_viz_pub.publish(viz_msg)

        # Publish PositionTarget
        target_msg = PositionTarget()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = cfg.BODY_FRAME_ID
        target_msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        target_msg.type_mask = (
            PositionTarget.IGNORE_PX |
            PositionTarget.IGNORE_PY |
            PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_VX |
            PositionTarget.IGNORE_VY |
            PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_YAW
        )
        target_msg.acceleration_or_force.x = filtered_vel[0]
        target_msg.acceleration_or_force.y = filtered_vel[1]
        target_msg.acceleration_or_force.z = filtered_vel[2]
        target_msg.yaw_rate = filtered_vel[3]
        self.local_setpoint_pub.publish(target_msg)
    
    # def pointcloud_callback(self, msg):
    #     """Process incoming point cloud (if using lidar instead of depth image)"""
    #     start_time = time.time()
    #     points3d_np = parse_pointcloud_optimized(msg)
    #     bins, bins_downsampled = binning_and_downsampling(points3d_np)
    #     bins_downsampled[bins_downsampled > cfg.LIDAR_MAX_RANGE] = cfg.LIDAR_MAX_RANGE

    #     self.lidar_tensor[:] = bins_downsampled.flatten().to(self.device)
    #     self.lidar_tensor[:] = 1 / self.lidar_tensor


    #     # Prepare observation on GPU (returns GPU tensor)
    #     obs_tensor_gpu = self.prepare_observation()
    #     obs_dict = {
    #         "observations": obs_tensor_gpu.unsqueeze(0)
    #     }
    #     with torch.no_grad():
    #         # Get action from network (input is GPU tensor, output is numpy on CPU)
    #         action = torch.clamp(self.policy.get_action(obs_dict), -1.0, 1.0).cpu().numpy().squeeze()
        
    #     # Publish action
    #     self.publish_action(action)
    #     print("PointCloud2 callback processing time: %.3f ms" % ((time.time() - start_time)*1000))


    def processed_lidar_callback(self, msg):
        """Consume pre-processed lidar from C++ node"""
        start_time = time.time()
        
        # C++ node publishes row-major Float32MultiArray
        # Dimension: [rows, cols] = [16, 20]
        # Data is either min_range or inverted depth (1/range)
        
        # Convert data to tensor and move to device
        lidar_data = torch.tensor(msg.data, device=self.device, dtype=torch.float32)
        
        # If the C++ node already inverted depth, we don't need to do it here
        # The Python node expects lidars to be inverted (1/distance)
        # Looking at the C++ code, it supports invert_depth_ parameter
        # If not inverted, we do: self.lidar_tensor[:] = 1.0 / (lidar_data + 1e-6)
        # Assuming for now it's matching the expected format
        self.lidar_tensor[:] = lidar_data.flatten()
        
        # Prepare observation on GPU
        obs_tensor_gpu = self.prepare_observation()
        obs_dict = {
            "observations": obs_tensor_gpu.unsqueeze(0)
        }
        
        with torch.no_grad():
            # Get action (input GPU, output CPU)
            action = torch.clamp(self.policy.get_action(obs_dict), -1.0, 1.0).cpu().numpy().squeeze()
        
        # Publish action
        self.publish_action(action)
        
        if (time.time() - start_time) * 1000 > 50: # Log only if slow
             self.get_logger().info("Processed Lidar callback time: %.3f ms" % ((time.time() - start_time)*1000))

    def image_callback(self, msg):
        """Main control loop triggered by image"""
        if not self.enable and cfg.USE_MAVROS_STATE:
            # Publish zero command
            self.publish_action(np.array([-1.0, 0.0, 0.0, 0.0]))
            return
        
        # Convert ROS Image to numpy
        if msg.encoding == "32FC1":  # Simulation
            depth_image = np.array(struct.unpack("f" * msg.height * msg.width, msg.data))
            depth_image = depth_image.reshape((msg.height, msg.width))
            depth_image[np.isnan(depth_image)] = cfg.IMAGE_MAX_DEPTH
        else:  # Real camera
            depth_image = np.ndarray((msg.height, msg.width), "<H", msg.data, 0)
            depth_image = depth_image.astype('float32') * 0.001
            depth_image[np.isnan(depth_image)] = cfg.IMAGE_MAX_DEPTH
        
        # Process image on GPU (returns GPU tensor)
        self.lidar_tensor = self.process_depth_image(depth_image)
        
        # Prepare observation on GPU (returns GPU tensor)
        obs_tensor_gpu = self.prepare_observation()
        
        # Get action from network (input is GPU tensor, output is numpy on CPU)
        action = self.policy.get_action(obs_tensor_gpu, normalize=True)
        
        # Store for next iteration
        self.prev_action = action
        
        # Publish action
        self.publish_action(action)

if __name__ == "__main__":
    import argparse
    
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--checkpoint", type=str, required=True,
    #                 #    help="Path to Sample Factory checkpoint")
    # parser.add_argument("--device", type=str, default="cuda:0",
    #                    help="Device to use for inference (cuda:0, cuda:1, cpu, etc.)")
    # args, _ = parser.parse_known_args()
    
    rclpy.init()
    node = LidarNavigationNode("cuda:0")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()