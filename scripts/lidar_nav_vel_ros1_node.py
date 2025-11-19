#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from mavros_msgs.msg import State
import ros_numpy
import time
import cv2
import numpy as np
import torch
import struct
from scipy.spatial.transform import Rotation as R

# Sample Factory inference (your standalone file)
from standalone_inference import SF_model_initializer


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
    ACTION_TOPIC = "/rmf/cmd/vel"
    TARGET_TOPIC = "/target"
    MAVROS_STATE_TOPIC = "/mavros/state"
    
    # Action transformation (match your training config)
    # These should match the action_transformation_function in your task config
    SPEED_SCALE = 1.5
    YAW_RATE_SCALE = 1.0
    
    # Frame IDs
    BODY_FRAME_ID = "mimosa_body"
    
    # Control
    USE_MAVROS_STATE = False
    ACTION_FILTER_ALPHA = 0.1  # EMA filter
    
    # Device
    DEVICE = "cuda:0"  # Default device, can be overridden by command line arg

    # Lidar
    LIDAR_MAX_RANGE = 10.0
    LIDAR_MIN_RANGE = 01.0

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
    spherical_points = torch.from_numpy(np.stack([r, theta, phi], axis=1))  # (N, 3)
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
    input_tensor = torch.full((azimuth_bins * elevation_bins,), 50.0)
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
        bins = bins.view(azimuth_bins, elevation_bins)
        bins2 = bins.T.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
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
    pc = ros_numpy.numpify(msg) #pc.shape=(720,1280)
    height = pc.shape[0]
    width = pc.shape[1]
    points3d_np = np.zeros((height * width, 3), dtype=np.float32)
    points3d_np[:, 0] = np.resize(pc['x'], height * width)
    points3d_np[:, 1] = np.resize(pc['y'], height * width)
    points3d_np[:, 2] = np.resize(pc['z'], height * width)
    # field_offsets = {field.name: field.offset for field in msg.fields}
    # x_offset = field_offsets.get('x', None)
    # y_offset = field_offsets.get('y', None)
    # z_offset = field_offsets.get('z', None)
    
    # if x_offset is None or y_offset is None or z_offset is None:
    #     raise ValueError("PointCloud2 must have x, y, z fields")
    
    # point_step = msg.point_step
    
    # # Check if xyz are packed at the beginning (most common case)
    # if x_offset == 0 and y_offset == 4 and z_offset == 8:
    #     # Ultra-fast path for standard layout
    #     num_floats_per_point = point_step // 4
    #     points3d_np = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, num_floats_per_point)[:, :3].copy()
    # else:
    #     print("Non-standard PointCloud2 layout detected, using slower parsing method.")
    #     # Fast path for non-standard layouts using numpy strides
    #     num_points = len(msg.data) // point_step
    #     data = np.frombuffer(msg.data, dtype=np.uint8)
        
    #     # Use stride tricks for efficient extraction
    #     points3d_np = np.zeros((num_points, 3), dtype=np.float32)
        
    #     for idx, offset in enumerate([x_offset, y_offset, z_offset]):
    #         byte_indices = np.arange(num_points)[:, None] * point_step + offset + np.arange(4)
    #         points3d_np[:, idx] = data[byte_indices].view(np.float32).flatten()
    # with open("points3d_xyz.csv", "w") as f:
    #     for point in points3d_np:
    #         f.write(f"{point[0]},{point[1]},{point[2]}\n")
    return points3d_np


# ============================================================================
# Main ROS Node
# ============================================================================
class LidarNavigationNode:
    def __init__(self, device="cuda:0"):
        rospy.init_node('lidar_navigation_node')
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load Sample Factory model on specified device
        self.policy = SF_model_initializer()

        # State variables
        self.position = np.zeros(3)
        self.target_position = np.zeros(3)
        self.target_position[0] = 5.0
        self.target_position[2] = 2.0
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
        
        # Publishers
        self.action_pub = rospy.Publisher(cfg.ACTION_TOPIC, Twist, queue_size=1)
        self.action_viz_pub = rospy.Publisher(cfg.ACTION_TOPIC + "_viz", TwistStamped, queue_size=1)
        self.filtered_action_pub = rospy.Publisher(cfg.ACTION_TOPIC + "_filtered", Twist, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber(cfg.IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        self.pointcloud_sub = rospy.Subscriber(cfg.POINTCLOUD_TOPIC, PointCloud2, self.pointcloud_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(cfg.ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)
        self.target_sub = rospy.Subscriber(cfg.TARGET_TOPIC, PoseStamped, self.target_callback, queue_size=1)
        self.reset_sub = rospy.Subscriber("/reset", Empty, self.reset_callback, queue_size=1)
        
        if cfg.USE_MAVROS_STATE:
            self.state_sub = rospy.Subscriber(cfg.MAVROS_STATE_TOPIC, State, self.state_callback, queue_size=1)
        
        rospy.loginfo("Lidar Navigation Node initialized")
    
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
    
    def target_callback(self, msg):
        """Update target position"""
        self.target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        # Reset on new target
        self.policy.reset()
        self.prev_action = np.zeros(cfg.ACTION_DIM)
        self.action_filter.reset()
        rospy.loginfo(f"New target: {self.target_position}")
    
    def reset_callback(self, msg):
        """Reset network state"""
        self.policy.reset()
        self.prev_action = np.zeros(cfg.ACTION_DIM)
        self.action_filter.reset()
        rospy.loginfo("Network reset")
    
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
        clamped_dist = np.clip(dist_to_target, 0.0, 5.0)
        
        # Unit vector to target
        unit_vec_to_target = vec_to_target_vehicle / (dist_to_target + 1e-6)
        
        # Fill state observation on CPU first (17 dims)
        self.state_obs_cpu[0:3] = torch.from_numpy(unit_vec_to_target).float()
        self.state_obs_cpu[3] = clamped_dist
        self.state_obs_cpu[4] = self.rpy[0]  # roll
        self.state_obs_cpu[5] = self.rpy[1]  # pitch
        self.state_obs_cpu[6] = ssa(0.0 - self.rpy[2])  # yaw to target (desired yaw is 0 in vehicle frame)
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
        vel_x = -(action[0] + 1.0) / 2.0 * cfg.SPEED_SCALE  # Forward is negative X
        # vel_x = action[0] * cfg.SPEED_SCALE
        vel_y = action[1] * cfg.SPEED_SCALE
        vel_z = action[2] * cfg.SPEED_SCALE
        yaw_rate = action[3] * cfg.YAW_RATE_SCALE
        
        return np.array([vel_x, vel_y, vel_z, yaw_rate])

    
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
        viz_msg.header.stamp = rospy.Time.now()
        viz_msg.header.frame_id = cfg.BODY_FRAME_ID
        viz_msg.twist = twist_msg
        self.action_viz_pub.publish(viz_msg)
    
    def pointcloud_callback(self, msg):
        """Process incoming point cloud (if using lidar instead of depth image)"""
        start_time = time.time()
        points3d_np = parse_pointcloud_optimized(msg)
        bins, bins_downsampled = binning_and_downsampling(points3d_np)
        bins_downsampled[bins_downsampled > cfg.LIDAR_MAX_RANGE] = cfg.LIDAR_MAX_RANGE

        # save bins to a csv file
        bins_np = bins.cpu().numpy()
        np.savetxt("bins.csv", bins_np.T.squeeze(), delimiter=",")
        np.savetxt("bins_downsampled.csv", bins_downsampled.cpu().numpy().squeeze(), delimiter=",")

        self.lidar_tensor[:] = bins_downsampled.flatten().to(self.device)
        self.lidar_tensor[:] = 1 / self.lidar_tensor


        # Prepare observation on GPU (returns GPU tensor)
        obs_tensor_gpu = self.prepare_observation()
        obs_dict = {
            "observations": obs_tensor_gpu.unsqueeze(0)
        }
        with torch.no_grad():
            # Get action from network (input is GPU tensor, output is numpy on CPU)
            action = torch.clamp(self.policy.get_action(obs_dict), -1.0, 1.0).cpu().numpy().squeeze()
        
        # Publish action
        self.publish_action(action)
        print("PointCloud2 callback processing time: %.3f ms" % ((time.time() - start_time)*1000))


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
    
    try:
        node = LidarNavigationNode("cuda:0")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass