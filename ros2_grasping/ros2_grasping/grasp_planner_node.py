#!/usr/bin/env python3
"""ROS 2 Node for running ICG-Net Grasp Planning with end-effector camera point clouds."""

import json
import struct
import sys
from pathlib import Path

import numpy as np

FILE_PATH = Path(__file__).resolve()


def find_workspace_root():
    for candidate in (Path.cwd().resolve(), *FILE_PATH.parents):
        if (candidate / 'scripts').is_dir() and (candidate / 'ros2_grasping').is_dir():
            return candidate
    return Path.cwd().resolve()


WORKSPACE_ROOT = find_workspace_root()

sys.path.insert(0, str(WORKSPACE_ROOT / 'third_party' / 'icg_net'))
sys.path.insert(0, str(WORKSPACE_ROOT / 'third_party' / 'icg_benchmark'))

import open3d as o3d
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from omegaconf import OmegaConf
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from std_srvs.srv import Trigger

try:
    from icg_net import ICGNetModule
    from icg_benchmark.grasping.planners import ICGNetPlanner
    from icg_benchmark.utils.timing.timer import Timer
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as R
except ImportError as e:
    print(f"Error importing grasp planner dependencies: {e}")
    sys.exit(1)


class GraspPlannerNode(Node):
    def __init__(self):
        super().__init__('grasp_planner_node')
        self.get_logger().info("Initializing Grasp Planner Node...")

        model_dir = (
            WORKSPACE_ROOT / 'third_party' / 'icg_benchmark' / 'data' /
            'icgnet' / '51--0.656'
        )
        default_config = str(model_dir / 'config.yaml')
        default_checkpoint = str(model_dir / 'checkpoint.ckpt')

        self.declare_parameter('config_path', default_config)
        self.declare_parameter('checkpoint_path', default_checkpoint)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('max_gripper_width', 0.08)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('voxel_size', 0.0045)
        self.declare_parameter('command_tcp_directly', True)
        # TCP offset relative to hand link flange (simulation hand_rotational.urdf values as default)
        self.declare_parameter('tcp_offset_x', -0.01)
        self.declare_parameter('tcp_offset_y', 0.0)
        self.declare_parameter('tcp_offset_z', 0.121)
        self.declare_parameter('gripper_close_value', 50)
        self.declare_parameter('gripper_open_value', 100)
        self.declare_parameter('camera_pose_mode', 'fixed')
        self.declare_parameter('fixed_camera_x', 0.490)
        self.declare_parameter('fixed_camera_y', 0.586)
        self.declare_parameter('fixed_camera_z', 0.489)
        self.declare_parameter('fixed_camera_roll', 1.0)
        self.declare_parameter('fixed_camera_pitch', 34.7)
        self.declare_parameter('fixed_camera_yaw', -99.5)
        self.declare_parameter('camera_offset_x', -0.160)
        self.declare_parameter('camera_offset_y', 0.060)
        self.declare_parameter('camera_offset_z', 0.060)
        self.declare_parameter('table_crop_min_x', 0.2)
        self.declare_parameter('table_crop_min_y', -0.5)
        self.declare_parameter('table_crop_min_z', -0.01)
        self.declare_parameter('table_crop_max_x', 0.9)
        self.declare_parameter('table_crop_max_y', 0.5)
        self.declare_parameter('table_crop_max_z', 0.2)
        self.declare_parameter('save_debug_pointclouds', False)
        self.declare_parameter('debug_pointcloud_dir', str(WORKSPACE_ROOT / 'debug' / 'pointclouds'))
        self.declare_parameter('save_debug_pointcloud_npz', True)
        self.declare_parameter('save_debug_pointcloud_ply', True)
        self.declare_parameter('geometric_target_object_index', 0)
        self.declare_parameter('geometric_target_min_height', 0.025)
        self.declare_parameter('geometric_target_max_height', 0.18)
        self.declare_parameter('pointcloud_origin_offset_x', 0.060)
        self.declare_parameter('pointcloud_origin_offset_y', 0.0)
        self.declare_parameter('pointcloud_origin_offset_z', 0.0)

        # Retrieve parameter values
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested but not available; falling back to CPU.")
            self.device = 'cpu'
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.max_gripper_width = self.get_parameter('max_gripper_width').get_parameter_value().double_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.command_tcp_directly = self.get_parameter('command_tcp_directly').get_parameter_value().bool_value
        self.tcp_offset_x = self.get_parameter('tcp_offset_x').get_parameter_value().double_value
        self.tcp_offset_y = self.get_parameter('tcp_offset_y').get_parameter_value().double_value
        self.tcp_offset_z = self.get_parameter('tcp_offset_z').get_parameter_value().double_value
        self.gripper_close_value = self.get_parameter('gripper_close_value').get_parameter_value().integer_value
        self.gripper_open_value = self.get_parameter('gripper_open_value').get_parameter_value().integer_value
        self.camera_pose_mode = self.get_parameter('camera_pose_mode').value
        if self.camera_pose_mode not in ('fixed', 'end_effector'):
            raise ValueError("camera_pose_mode must be 'fixed' or 'end_effector'")
        self.fixed_camera_position_m = np.array([
            self.get_parameter('fixed_camera_x').value,
            self.get_parameter('fixed_camera_y').value,
            self.get_parameter('fixed_camera_z').value,
        ], dtype=np.float64)
        self.fixed_camera_rpy_deg = np.array([
            self.get_parameter('fixed_camera_roll').value,
            self.get_parameter('fixed_camera_pitch').value,
            self.get_parameter('fixed_camera_yaw').value,
        ], dtype=np.float64)
        self.camera_offset_from_ee_m = np.array([
            self.get_parameter('camera_offset_x').value,
            self.get_parameter('camera_offset_y').value,
            self.get_parameter('camera_offset_z').value,
        ], dtype=np.float64)
        self.table_crop_min_base_m = np.array([
            self.get_parameter('table_crop_min_x').value,
            self.get_parameter('table_crop_min_y').value,
            self.get_parameter('table_crop_min_z').value,
        ], dtype=np.float64)
        self.table_crop_max_base_m = np.array([
            self.get_parameter('table_crop_max_x').value,
            self.get_parameter('table_crop_max_y').value,
            self.get_parameter('table_crop_max_z').value,
        ], dtype=np.float64)
        self.save_debug_pointclouds = self.get_parameter('save_debug_pointclouds').get_parameter_value().bool_value
        self.debug_pointcloud_dir = Path(
            self.get_parameter('debug_pointcloud_dir').get_parameter_value().string_value
        ).expanduser()
        self.save_debug_pointcloud_npz = (
            self.get_parameter('save_debug_pointcloud_npz').get_parameter_value().bool_value
        )
        self.save_debug_pointcloud_ply = (
            self.get_parameter('save_debug_pointcloud_ply').get_parameter_value().bool_value
        )
        self.geometric_target_object_index = (
            self.get_parameter('geometric_target_object_index').get_parameter_value().integer_value
        )
        self.geometric_target_min_height = (
            self.get_parameter('geometric_target_min_height').get_parameter_value().double_value
        )
        self.geometric_target_max_height = (
            self.get_parameter('geometric_target_max_height').get_parameter_value().double_value
        )
        self.pointcloud_origin_offset_m = np.array([
            self.get_parameter('pointcloud_origin_offset_x').get_parameter_value().double_value,
            self.get_parameter('pointcloud_origin_offset_y').get_parameter_value().double_value,
            self.get_parameter('pointcloud_origin_offset_z').get_parameter_value().double_value,
        ], dtype=np.float64)

        self.get_logger().info(f"Config path: {self.config_path}")
        self.get_logger().info(f"Checkpoint path: {self.checkpoint_path}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Camera pose mode: {self.camera_pose_mode}")
        self.get_logger().info(
            "Pointcloud origin offset in camera frame [mm]: "
            f"{self.format_vector(self.pointcloud_origin_offset_m * 1000.0, 1)}"
        )

        # In-memory config and checkpoint loading
        if not Path(self.config_path).is_file():
            raise FileNotFoundError(f"Config path does not exist: {self.config_path}")
        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint path does not exist: {self.checkpoint_path}")

        # Load configuration
        self.get_logger().info("Loading model configuration...")
        cfg = OmegaConf.load(self.config_path)
        cfg.general.checkpoint = str(Path(self.checkpoint_path).resolve())

        # Load ICG-Net model
        self.get_logger().info("Loading model checkpoint weights...")
        self.model = ICGNetModule(
            config=cfg,
            device=self.device,
            grasp_each_object=True,
            n_grasps=8192,
            n_grasp_pred_orientations=6,
            gripper_offset=0.0,
            gripper_offset_perc=10.5,
            max_gripper_width=self.max_gripper_width,
            full_width=True,
            coll_checks=True,
        ).eval()

        # Load grasp planner
        self.planner = ICGNetPlanner(
            self.model,
            device=self.device,
            confidence_th=self.confidence_threshold,
            resample=False,
            visualize=False,
            latent_imagination=False,
            use_fps=True
        )
        self.get_logger().info("Model and Planner successfully initialized.")

        # Cache for latest point cloud message
        self.latest_cloud_msg = None
        self.latest_cart_pos = None
        self.last_cropped_points_world_m = None
        self.logged_instance_name_note = False
        self.saved_pointcloud_count = 0
        if self.save_debug_pointclouds:
            self.debug_pointcloud_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Pointcloud capture enabled: {self.debug_pointcloud_dir}")

        # Publishers
        self.mode_pub = self.create_publisher(Int32MultiArray, '/FRT/arm/L0/mode/set', 10)
        self.payload_pub = self.create_publisher(Int32MultiArray, '/FRT/arm/L0/payload/set', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/grasp_pose', 10)

        # Subscriptions
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/FRT/SW/track/zed/hand/pointcloud',
            self.pointcloud_callback,
            10
        )
        cart_pos_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cart_pos_sub = self.create_subscription(
            Float32MultiArray,
            '/FRT/arm/L0/cartPos',
            self.cart_pos_callback,
            cart_pos_qos
        )

        # Services
        self.plan_srv = self.create_service(Trigger, '/plan_grasp', self.plan_grasp_callback)
        self.open_srv = self.create_service(Trigger, '/open_gripper', self.open_gripper_callback)
        self.close_srv = self.create_service(Trigger, '/close_gripper', self.close_gripper_callback)

        # Keep track of last commanded pose (for open/close services)
        self.last_x = 0
        self.last_y = 0
        self.last_z = 0
        self.last_roll = 0
        self.last_pitch = 0
        self.last_yaw = 0

        self.get_logger().info("Grasp Planner Node is ready.")

    def pointcloud_callback(self, msg: PointCloud2):
        self.latest_cloud_msg = msg
        if self.save_debug_pointclouds:
            self.save_pointcloud_capture(msg)

    def cart_pos_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 6:
            self.get_logger().warn(f"Ignoring invalid cartPos message with {len(msg.data)} values.")
            return

        cart_pos = np.array(msg.data[:6], dtype=np.float64)
        if not np.all(np.isfinite(cart_pos)):
            self.get_logger().warn(f"Ignoring non-finite cartPos message: {msg.data[:6]}")
            return

        self.latest_cart_pos = cart_pos

    def plan_grasp_callback(self, request, response):
        self.get_logger().info("Received grasp planning request.")

        # Check if we have received a pointcloud
        cloud_msg = self.latest_cloud_msg
        if cloud_msg is None:
            self.get_logger().warn("No pointcloud received yet on /FRT/SW/track/zed/hand/pointcloud")
            response.success = False
            response.message = "No pointcloud received yet."
            return response

        cart_pos = self.latest_cart_pos
        if cart_pos is None:
            self.get_logger().warn("No arm Cartesian pose received yet on /FRT/arm/L0/cartPos")
            response.success = False
            response.message = "No arm Cartesian pose received yet."
            return response

        # Build transforms from the current end-effector Cartesian pose.
        # /FRT/arm/L0/cartPos is the end-effector pose in world:
        # [x, y, z, roll, pitch, yaw], with xyz in millimeters and rpy in degrees.
        T_ee_to_base = self.cart_pos_to_matrix(cart_pos)
        T_cam_to_base = self.camera_pose_from_ee_pose(T_ee_to_base)
        self.get_logger().info(f"Current arm Cartesian pose [mm, deg]: {cart_pos.tolist()}")
        self.get_logger().info(
            "Camera pose in base frame: "
            f"position_mm={self.format_vector(T_cam_to_base[:3, 3] * 1000.0, 1)}, "
            f"mode={self.camera_pose_mode}"
        )

        # Decode pointcloud message to Open3D pointcloud
        self.get_logger().info("Decoding PointCloud2 message...")
        pc = self.decode_pointcloud(cloud_msg, T_cam_to_base)
        if pc is None:
            self.get_logger().warn("Decoded pointcloud is empty or has too few points.")
            response.success = False
            response.message = "Decoded pointcloud is empty."
            return response

        # Run ICG-Net grasp planner
        self.get_logger().info("Running grasp planner model inference...")
        timer = Timer()

        def get_img(t):
            return pc

        out = self.planner(get_img, timer)
        planner_debug = self.log_planner_prediction_summary()
        planner_debug.update(self.find_geometric_object_targets())

        if out is None:
            self.get_logger().warn("No valid grasp candidates found above threshold.")
            response.success = False
            response.message = "No valid grasp candidates found."
            return response

        grasp_pose_icg = out[0][0]  # shape (4, 4) numpy array in ICG/world pointcloud frame
        self.get_logger().info(f"Planned grasp pose in ICG/world frame:\n{grasp_pose_icg}")

        # ICG-Net expects a table/world frame with +Z as height, so decode_pointcloud()
        # feeds it cropped base-frame points. Its output pose is therefore already in base.
        grasp_pose_base = grasp_pose_icg
        self.get_logger().info(f"Grasp pose in base frame:\n{grasp_pose_base}")

        object_target_base = self.select_object_target_in_base(planner_debug, T_cam_to_base, grasp_pose_base)
        command_pose_base = np.eye(4)
        command_pose_base[:3, :3] = T_ee_to_base[:3, :3]
        command_pose_base[:3, 3] = object_target_base
        self.get_logger().info(
            "Using selected object target with current end-effector orientation; "
            "ICG-Net gripper orientation is logged but not sent to SFIGA."
        )

        # Publish the command pose to /grasp_pose for visualization.
        self.publish_grasp_pose_vis(command_pose_base)

        # Convert pose components
        x_mm = command_pose_base[0, 3] * 1000.0
        y_mm = command_pose_base[1, 3] * 1000.0
        z_mm = command_pose_base[2, 3] * 1000.0

        r = R.from_matrix(command_pose_base[:3, :3])
        roll_deg, pitch_deg, yaw_deg = r.as_euler('xyz', degrees=True)

        self.last_x = int(round(x_mm))
        self.last_y = int(round(y_mm))
        self.last_z = int(round(z_mm))
        self.last_roll = int(round(roll_deg))
        self.last_pitch = int(round(pitch_deg))
        self.last_yaw = int(round(yaw_deg))

        # Command the robot arm
        self.log_selected_grasp_plan(
            grasp_pose_icg,
            grasp_pose_base,
            command_pose_base,
            planner_debug,
            T_ee_to_base,
            T_cam_to_base,
        )
        self.publish_arm_commands(self.last_x, self.last_y, self.last_z,
                                  self.last_roll, self.last_pitch, self.last_yaw,
                                  self.gripper_open_value)

        response.success = True
        response.message = (
            "Grasp planned and payload published. "
            f"Target: x={self.last_x}mm, y={self.last_y}mm, z={self.last_z}mm, "
            f"r={self.last_roll}deg, p={self.last_pitch}deg, y={self.last_yaw}deg"
        )
        return response

    def open_gripper_callback(self, request, response):
        self.get_logger().info("Received gripper open command.")
        self.publish_arm_commands(self.last_x, self.last_y, self.last_z,
                                  self.last_roll, self.last_pitch, self.last_yaw,
                                  self.gripper_open_value)
        response.success = True
        response.message = f"Gripper OPEN payload published. End-effector value: {self.gripper_open_value}"
        return response

    def close_gripper_callback(self, request, response):
        self.get_logger().info("Received gripper close command.")
        self.publish_arm_commands(self.last_x, self.last_y, self.last_z,
                                  self.last_roll, self.last_pitch, self.last_yaw,
                                  self.gripper_close_value)
        response.success = True
        response.message = f"Gripper CLOSE payload published. End-effector value: {self.gripper_close_value}"
        return response

    def decode_pointcloud(self, msg: PointCloud2, T_cam_to_base: np.ndarray) -> o3d.geometry.PointCloud | None:
        try:
            points = self.pointcloud_msg_to_xyz_array(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode PointCloud2 xyz fields: {e}")
            return None

        if len(points) == 0:
            self.get_logger().warn(
                f"PointCloud2 decoded from frame '{msg.header.frame_id}' but contained no finite xyz points."
            )
            return None

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        raw_min = points.min(axis=0)
        raw_max = points.max(axis=0)
        self.get_logger().info(
            "PointCloud2 decoded: "
            f"frame='{msg.header.frame_id}', stamp={stamp_sec:.3f}, "
            f"raw_points={len(points)}, bounds_m=min{self.format_vector(raw_min, 3)} "
            f"max{self.format_vector(raw_max, 3)}"
        )

        points_camera = self.crop_points_to_table_workspace(points, T_cam_to_base)
        if len(points_camera) == 0:
            self.get_logger().warn(
                "Pointcloud rejected by table crop: no points remained inside "
                f"world min{self.format_vector(self.table_crop_min_base_m, 3)} "
                f"max{self.format_vector(self.table_crop_max_base_m, 3)}"
            )
            return None

        self.last_cropped_points_world_m = self.transform_points_to_base(
            points_camera,
            T_cam_to_base,
            apply_pointcloud_offset=False,
        ).astype(np.float32)
        points = self.last_cropped_points_world_m.astype(np.float64)

        # Build Open3D pointcloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        # Remove statistical/radius outliers
        pc, ind = pc.remove_radius_outlier(nb_points=20, radius=0.02)
        filtered_count = len(pc.points)
        self.get_logger().info(
            "Pointcloud radius filtering: "
            f"kept={filtered_count}/{len(points)}, removed={len(points) - filtered_count}, "
            "nb_points=20, radius=0.020m"
        )
        if len(pc.points) < 50:
            self.get_logger().warn(
                f"Pointcloud rejected after filtering: kept={len(pc.points)} points, minimum required=50."
            )
            return None

        # Estimate normals and orient towards the camera location in the same base/world frame.
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        pc.orient_normals_towards_camera_location(camera_location=T_cam_to_base[:3, 3])

        # Downsample pointcloud
        if self.voxel_size > 0:
            before_downsample = len(pc.points)
            pc = pc.voxel_down_sample(voxel_size=self.voxel_size)
            self.get_logger().info(
                "Pointcloud voxel downsample: "
                f"voxel_size={self.voxel_size:.4f}m, points={before_downsample}->{len(pc.points)}"
            )

        self.get_logger().info(
            f"Pointcloud ready for ICG-Net in base/world frame: points={len(pc.points)}, normals_estimated=True"
        )

        return pc

    def apply_pointcloud_origin_offset(self, points_camera: np.ndarray) -> np.ndarray:
        return points_camera.astype(np.float64) + self.pointcloud_origin_offset_m

    def transform_points_to_base(
        self,
        points_camera: np.ndarray,
        T_cam_to_base: np.ndarray,
        apply_pointcloud_offset: bool = True,
    ) -> np.ndarray:
        rotation = T_cam_to_base[:3, :3]
        translation = T_cam_to_base[:3, 3]
        points = points_camera.astype(np.float64)
        if apply_pointcloud_offset:
            points = self.apply_pointcloud_origin_offset(points)
        return points @ rotation.T + translation

    def table_workspace_mask(self, points_base: np.ndarray) -> np.ndarray:
        lower = self.table_crop_min_base_m
        upper = self.table_crop_max_base_m
        return np.all((points_base >= lower) & (points_base <= upper), axis=1)

    def crop_points_to_table_workspace(self, points_camera: np.ndarray, T_cam_to_base: np.ndarray) -> np.ndarray:
        corrected_points_camera = self.apply_pointcloud_origin_offset(points_camera)
        points_base = self.transform_points_to_base(
            corrected_points_camera,
            T_cam_to_base,
            apply_pointcloud_offset=False,
        )
        mask = self.table_workspace_mask(points_base)
        lower = self.table_crop_min_base_m
        upper = self.table_crop_max_base_m
        kept_count = int(mask.sum())
        removed_count = int(len(points_camera) - kept_count)

        if kept_count > 0:
            kept_world = points_base[mask]
            self.get_logger().info(
                "Table workspace crop in world frame: "
                f"kept={kept_count}/{len(points_camera)}, removed={removed_count}, "
                f"crop_min_m={self.format_vector(lower, 3)}, "
                f"crop_max_m={self.format_vector(upper, 3)}, "
                f"kept_world_bounds_m=min{self.format_vector(kept_world.min(axis=0), 3)} "
                f"max{self.format_vector(kept_world.max(axis=0), 3)}"
            )
        else:
            self.get_logger().warn(
                "Table workspace crop in world frame kept zero points: "
                f"input_world_bounds_m=min{self.format_vector(points_base.min(axis=0), 3)} "
                f"max{self.format_vector(points_base.max(axis=0), 3)}, "
                f"crop_min_m={self.format_vector(lower, 3)}, "
                f"crop_max_m={self.format_vector(upper, 3)}"
            )

        return corrected_points_camera[mask]

    def find_geometric_object_targets(self):
        debug = {
            'table_plane_tilt_deg': None,
            'geometric_components_base_m': [],
            'selected_geometric_component': None,
            'selected_geometry_target_base_m': None,
        }

        points_base = self.last_cropped_points_world_m
        if points_base is None or len(points_base) < 100:
            self.get_logger().warn("No cropped world pointcloud available for geometric object target selection.")
            return debug

        try:
            normal, plane_offset, tilt_deg, residual_std = self.fit_table_plane(points_base)
        except Exception as e:
            self.get_logger().warn(f"Failed to fit table plane for geometric target selection: {e}")
            return debug

        heights = points_base @ normal + plane_offset
        object_mask = (
            (heights > self.geometric_target_min_height)
            & (heights < self.geometric_target_max_height)
        )
        object_points = points_base[object_mask]
        debug['table_plane_tilt_deg'] = float(tilt_deg)

        self.get_logger().info(
            "Geometric table plane: "
            f"normal={self.format_vector(normal, 4)}, "
            f"tilt={tilt_deg:.2f}deg, residual_std={residual_std * 1000.0:.1f}mm, "
            f"object_height_window_m=[{self.geometric_target_min_height:.3f}, "
            f"{self.geometric_target_max_height:.3f}], candidate_points={len(object_points)}"
        )

        components = self.extract_geometric_components(object_points)
        debug['geometric_components_base_m'] = components
        if not components:
            self.get_logger().warn("No above-table geometric object components found.")
            return debug

        entries = []
        for index, component in enumerate(components[:6]):
            center = np.asarray(component['center_base_m'], dtype=np.float64)
            size = np.asarray(component['size_base_m'], dtype=np.float64)
            entries.append(
                f"#{index}: points={component['point_count']} "
                f"center_mm={self.format_vector(center * 1000.0, 1)} "
                f"size_mm={self.format_vector(size * 1000.0, 1)}"
            )
        suffix = "" if len(components) <= 6 else f"; +{len(components) - 6} more"
        self.get_logger().info(
            f"Geometric above-table components: {'; '.join(entries)}{suffix}"
        )

        selected_index = int(np.clip(self.geometric_target_object_index, 0, len(components) - 1))
        selected = components[selected_index]
        target = np.asarray(selected['center_base_m'], dtype=np.float64)
        debug['selected_geometric_component'] = selected_index
        debug['selected_geometry_target_base_m'] = target
        self.get_logger().info(
            "Selected geometric object target: "
            f"component=#{selected_index}, "
            f"target_world_mm={self.format_vector(target * 1000.0, 1)}"
        )
        return debug

    def fit_table_plane(self, points_base: np.ndarray):
        low_z_cut = np.percentile(points_base[:, 2], 35.0)
        plane_points = points_base[points_base[:, 2] <= low_z_cut]
        if len(plane_points) > 120000:
            indices = np.linspace(0, len(plane_points) - 1, 120000, dtype=np.int64)
            plane_points = plane_points[indices]

        design = np.c_[plane_points[:, 0], plane_points[:, 1], np.ones(len(plane_points))]
        coeff, *_ = np.linalg.lstsq(design, plane_points[:, 2], rcond=None)
        normal = np.array([-coeff[0], -coeff[1], 1.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        if normal[2] < 0.0:
            normal *= -1.0

        offset = -float(np.dot(normal, plane_points.mean(axis=0)))
        residual = plane_points @ normal + offset
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
        residual_std = float(np.std(residual))
        return normal, offset, tilt_deg, residual_std

    def extract_geometric_components(self, points_base: np.ndarray):
        if len(points_base) == 0:
            return []

        voxel_size = 0.008
        neighbor_radius = 0.020
        min_component_points = 250

        voxel_keys = np.floor(points_base / voxel_size).astype(np.int64)
        unique_keys, inverse, counts = np.unique(
            voxel_keys,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        centroids = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(centroids, inverse, points_base)
        centroids /= counts[:, None]

        tree = cKDTree(centroids)
        neighbors = tree.query_ball_point(centroids, r=neighbor_radius)
        seen = np.zeros(len(centroids), dtype=bool)
        components = []

        for start_index in range(len(centroids)):
            if seen[start_index]:
                continue

            stack = [start_index]
            seen[start_index] = True
            members = []
            while stack:
                current = stack.pop()
                members.append(current)
                for neighbor in neighbors[current]:
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        stack.append(neighbor)

            member_indices = np.asarray(members, dtype=np.int64)
            point_count = int(counts[member_indices].sum())
            if point_count < min_component_points:
                continue

            member_centroids = centroids[member_indices]
            center = (
                member_centroids * counts[member_indices, None]
            ).sum(axis=0) / point_count
            bounds_min = member_centroids.min(axis=0)
            bounds_max = member_centroids.max(axis=0)
            size = bounds_max - bounds_min
            horizontal_size = max(float(np.linalg.norm(size[:2])), 0.03)
            score = point_count / horizontal_size

            components.append({
                'point_count': point_count,
                'center_base_m': center,
                'bounds_min_base_m': bounds_min,
                'bounds_max_base_m': bounds_max,
                'size_base_m': size,
                'score': float(score),
            })

        components.sort(key=lambda component: component['score'], reverse=True)
        return components

    def pointcloud_msg_to_xyz_array(self, msg: PointCloud2) -> np.ndarray:
        try:
            import sensor_msgs_py.point_cloud2 as pc2
            points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(points_gen), dtype=np.float32)
        except (ImportError, AttributeError, TypeError, ValueError):
            points = []
            x_offset = next(f.offset for f in msg.fields if f.name == 'x')
            y_offset = next(f.offset for f in msg.fields if f.name == 'y')
            z_offset = next(f.offset for f in msg.fields if f.name == 'z')
            for i in range(0, len(msg.data), msg.point_step):
                x, = struct.unpack_from('f', msg.data, i + x_offset)
                y, = struct.unpack_from('f', msg.data, i + y_offset)
                z, = struct.unpack_from('f', msg.data, i + z_offset)
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    points.append([x, y, z])
            points = np.array(points, dtype=np.float32)

        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return points[np.all(np.isfinite(points), axis=1)]

    def save_pointcloud_capture(self, msg: PointCloud2):
        try:
            points_camera = self.pointcloud_msg_to_xyz_array(msg)
            if len(points_camera) == 0:
                self.get_logger().warn("Skipping pointcloud capture: no finite xyz points.")
                return

            if self.latest_cart_pos is None:
                self.get_logger().warn(
                    "Skipping pointcloud capture: no /FRT/arm/L0/cartPos received yet, "
                    "so the cloud cannot be saved in world coordinates."
                )
                return

            T_ee_to_base = self.cart_pos_to_matrix(self.latest_cart_pos)
            T_cam_to_base = self.camera_pose_from_ee_pose(T_ee_to_base)
            corrected_points_camera = self.apply_pointcloud_origin_offset(points_camera)
            points_base = self.transform_points_to_base(points_camera, T_cam_to_base)
            table_mask = self.table_workspace_mask(points_base)
            cropped_points_base = points_base[table_mask].astype(np.float32)
            kept_count = int(len(cropped_points_base))
            removed_count = int(len(points_base) - kept_count)
            if kept_count == 0:
                self.get_logger().warn(
                    "Skipping pointcloud capture: table crop kept zero world-frame points. "
                    f"raw_world_bounds_m=min{self.format_vector(points_base.min(axis=0), 3)} "
                    f"max{self.format_vector(points_base.max(axis=0), 3)}, "
                    f"crop_min_m={self.format_vector(self.table_crop_min_base_m, 3)}, "
                    f"crop_max_m={self.format_vector(self.table_crop_max_base_m, 3)}"
                )
                return

            stamp_name = f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec:09d}"
            capture_index = self.saved_pointcloud_count
            self.saved_pointcloud_count += 1
            base_path = self.debug_pointcloud_dir / f"pointcloud_{capture_index:06d}_{stamp_name}"
            latest_base = self.debug_pointcloud_dir / "latest_pointcloud"

            saved_paths = []
            if self.save_debug_pointcloud_npz:
                npz_path = base_path.with_suffix(".npz")
                latest_npz_path = latest_base.with_suffix(".npz")
                np.savez_compressed(
                    npz_path,
                    pc=cropped_points_base,
                    points_world_m=cropped_points_base,
                    frame_id=np.array(self.base_frame),
                    source_frame_id=np.array(msg.header.frame_id),
                    coordinate_frame=np.array("world/base"),
                    cropped_to_table_workspace=np.array(True),
                    table_crop_min_base_m=self.table_crop_min_base_m.astype(np.float32),
                    table_crop_max_base_m=self.table_crop_max_base_m.astype(np.float32),
                    pointcloud_origin_offset_m=self.pointcloud_origin_offset_m.astype(np.float32),
                    T_cam_to_base=T_cam_to_base.astype(np.float64),
                    T_ee_to_base=T_ee_to_base.astype(np.float64),
                    raw_point_count=np.array(len(points_camera), dtype=np.int64),
                    crop_removed_count=np.array(removed_count, dtype=np.int64),
                    stamp_sec=np.array(msg.header.stamp.sec, dtype=np.int64),
                    stamp_nanosec=np.array(msg.header.stamp.nanosec, dtype=np.int64),
                )
                np.savez_compressed(
                    latest_npz_path,
                    pc=cropped_points_base,
                    points_world_m=cropped_points_base,
                    frame_id=np.array(self.base_frame),
                    source_frame_id=np.array(msg.header.frame_id),
                    coordinate_frame=np.array("world/base"),
                    cropped_to_table_workspace=np.array(True),
                    table_crop_min_base_m=self.table_crop_min_base_m.astype(np.float32),
                    table_crop_max_base_m=self.table_crop_max_base_m.astype(np.float32),
                    pointcloud_origin_offset_m=self.pointcloud_origin_offset_m.astype(np.float32),
                    T_cam_to_base=T_cam_to_base.astype(np.float64),
                    T_ee_to_base=T_ee_to_base.astype(np.float64),
                    raw_point_count=np.array(len(points_camera), dtype=np.int64),
                    crop_removed_count=np.array(removed_count, dtype=np.int64),
                    stamp_sec=np.array(msg.header.stamp.sec, dtype=np.int64),
                    stamp_nanosec=np.array(msg.header.stamp.nanosec, dtype=np.int64),
                )
                saved_paths.append(npz_path)

            if self.save_debug_pointcloud_ply:
                ply_path = base_path.with_suffix(".ply")
                latest_ply_path = latest_base.with_suffix(".ply")
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(cropped_points_base)
                o3d.io.write_point_cloud(str(ply_path), pc, write_ascii=False, compressed=False)
                o3d.io.write_point_cloud(str(latest_ply_path), pc, write_ascii=False, compressed=False)
                saved_paths.append(ply_path)

            metadata = {
                "frame_id": self.base_frame,
                "source_frame_id": msg.header.frame_id,
                "coordinate_frame": "world/base",
                "cropped_to_table_workspace": True,
                "stamp_sec": int(msg.header.stamp.sec),
                "stamp_nanosec": int(msg.header.stamp.nanosec),
                "point_count": int(kept_count),
                "raw_point_count": int(len(points_camera)),
                "crop_removed_count": int(removed_count),
                "bounds_min_m": cropped_points_base.min(axis=0).astype(float).tolist(),
                "bounds_max_m": cropped_points_base.max(axis=0).astype(float).tolist(),
                "raw_camera_bounds_min_m": points_camera.min(axis=0).astype(float).tolist(),
                "raw_camera_bounds_max_m": points_camera.max(axis=0).astype(float).tolist(),
                "corrected_camera_bounds_min_m": corrected_points_camera.min(axis=0).astype(float).tolist(),
                "corrected_camera_bounds_max_m": corrected_points_camera.max(axis=0).astype(float).tolist(),
                "raw_world_bounds_min_m": points_base.min(axis=0).astype(float).tolist(),
                "raw_world_bounds_max_m": points_base.max(axis=0).astype(float).tolist(),
                "table_crop_min_base_m": self.table_crop_min_base_m.astype(float).tolist(),
                "table_crop_max_base_m": self.table_crop_max_base_m.astype(float).tolist(),
                "pointcloud_origin_offset_m": self.pointcloud_origin_offset_m.astype(float).tolist(),
                "camera_position_world_m": T_cam_to_base[:3, 3].astype(float).tolist(),
                "camera_rpy_world_deg": R.from_matrix(T_cam_to_base[:3, :3]).as_euler('xyz', degrees=True).astype(float).tolist(),
                "end_effector_position_world_m": T_ee_to_base[:3, 3].astype(float).tolist(),
                "end_effector_rpy_world_deg": R.from_matrix(T_ee_to_base[:3, :3]).as_euler('xyz', degrees=True).astype(float).tolist(),
                "fields": [field.name for field in msg.fields],
                "height": int(msg.height),
                "width": int(msg.width),
                "point_step": int(msg.point_step),
                "row_step": int(msg.row_step),
            }
            metadata_path = base_path.with_suffix(".json")
            latest_metadata_path = latest_base.with_suffix(".json")
            metadata_text = json.dumps(metadata, indent=2, sort_keys=True)
            metadata_path.write_text(metadata_text + "\n", encoding="utf-8")
            latest_metadata_path.write_text(metadata_text + "\n", encoding="utf-8")
            saved_paths.append(metadata_path)

            saved_list = ", ".join(str(path) for path in saved_paths)
            self.get_logger().info(
                "Saved cropped world pointcloud capture "
                f"#{capture_index} with {kept_count}/{len(points_camera)} points "
                f"(removed={removed_count}): {saved_list}"
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to save pointcloud capture: {e}")

    def to_numpy(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        try:
            return np.asarray(value)
        except (TypeError, ValueError):
            return None

    def format_vector(self, values, precision=3):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        return np.array2string(arr, precision=precision, separator=', ', suppress_small=False)

    def log_planner_prediction_summary(self):
        debug = {
            'selected_instance': None,
            'selected_score': None,
            'selected_width_m': None,
            'selected_contact_camera_m': None,
            'selected_gripper_axes_camera': None,
            'selected_object_center_camera_m': None,
            'selected_object_bounds_camera_m': None,
        }

        prediction = getattr(self.planner, 'embeddings', None)
        if prediction is None:
            self.get_logger().warn("ICG-Net did not expose prediction details for this planning run.")
            return debug

        if not self.logged_instance_name_note:
            self.get_logger().info(
                "ICG-Net output exposes object instance ids, not semantic object names; "
                "verbose logs report detected instances by id."
            )
            self.logged_instance_name_note = True

        self.log_detected_instances(prediction)

        scene_grasps = getattr(prediction, 'scene_grasp_poses', None)
        if not scene_grasps or len(scene_grasps) < 5:
            self.get_logger().warn("ICG-Net produced no scene grasp candidate tensors.")
            return debug

        orientation, contact, score, width, instance = scene_grasps
        score_np = self.to_numpy(score)
        if score_np is None:
            self.get_logger().warn("ICG-Net grasp scores could not be decoded for logging.")
            return debug

        score_np = score_np.reshape(-1)
        total_candidates = score_np.size
        if total_candidates == 0:
            self.get_logger().warn("ICG-Net produced zero grasp candidates.")
            return debug

        width_np = self.to_numpy(width)
        if width_np is None or width_np.size != total_candidates:
            width_np = np.full(total_candidates, np.nan)
        else:
            width_np = width_np.reshape(-1)

        instance_np = self.to_numpy(instance)
        if instance_np is None or instance_np.size != total_candidates:
            instance_np = np.full(total_candidates, -1)
        else:
            instance_np = instance_np.reshape(-1).astype(np.int64)

        contact_np = self.to_numpy(contact)
        if contact_np is not None and contact_np.size == total_candidates * 3:
            contact_np = contact_np.reshape(total_candidates, 3)
        else:
            contact_np = None

        orientation_np = self.to_numpy(orientation)
        if orientation_np is not None and orientation_np.size == total_candidates * 9:
            orientation_np = orientation_np.reshape(total_candidates, 3, 3)
        else:
            orientation_np = None

        valid_mask = score_np > self.confidence_threshold
        valid_count = int(valid_mask.sum())
        best_all_index = int(np.argmax(score_np))
        best_all_score = float(score_np[best_all_index])

        if valid_count == 0:
            self.get_logger().warn(
                "ICG-Net grasp candidates: "
                f"total={total_candidates}, above_threshold=0, "
                f"threshold={self.confidence_threshold:.3f}, best_score={best_all_score:.3f}, "
                f"best_instance={int(instance_np[best_all_index])}"
            )
            self.log_candidate_instances(score_np, width_np, instance_np, valid_mask)
            return debug

        valid_indices = np.flatnonzero(valid_mask)
        best_valid_index = int(valid_indices[np.argmax(score_np[valid_mask])])
        debug['selected_instance'] = int(instance_np[best_valid_index])
        debug['selected_score'] = float(score_np[best_valid_index])
        debug['selected_width_m'] = float(width_np[best_valid_index])

        if contact_np is not None:
            debug['selected_contact_camera_m'] = contact_np[best_valid_index]

        if orientation_np is not None:
            debug['selected_gripper_axes_camera'] = orientation_np[best_valid_index]

        center, bounds = self.extract_instance_geometry(prediction, debug['selected_instance'])
        debug['selected_object_center_camera_m'] = center
        debug['selected_object_bounds_camera_m'] = bounds

        self.get_logger().info(
            "ICG-Net grasp candidates: "
            f"total={total_candidates}, above_threshold={valid_count}, "
            f"threshold={self.confidence_threshold:.3f}, "
            f"selected_instance={debug['selected_instance']}, "
            f"selected_score={debug['selected_score']:.3f}, "
            f"selected_width={debug['selected_width_m'] * 1000.0:.1f}mm"
        )
        if debug['selected_contact_camera_m'] is not None:
            self.get_logger().info(
                "Selected contact point in ICG/world frame: "
                f"{self.format_vector(debug['selected_contact_camera_m'], 4)} m"
            )
        if debug['selected_object_center_camera_m'] is not None:
            self.get_logger().info(
                "Selected object instance center in ICG/world frame: "
                f"{self.format_vector(debug['selected_object_center_camera_m'], 4)} m"
            )
            bounds_min, bounds_max = debug['selected_object_bounds_camera_m']
            self.get_logger().info(
                "Selected object instance bounds in ICG/world frame: "
                f"min{self.format_vector(bounds_min, 4)} m "
                f"max{self.format_vector(bounds_max, 4)} m"
            )

        self.log_candidate_instances(score_np, width_np, instance_np, valid_mask)
        return debug

    def extract_instance_geometry(self, prediction, selected_instance):
        embedding = getattr(prediction, 'embedding', None)
        if embedding is None or selected_instance is None:
            return None, None

        points = self.to_numpy(getattr(embedding, 'voxelized_pc', None))
        labels = self.to_numpy(getattr(embedding, 'class_labels', None))
        if points is None or labels is None:
            return None, None

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(labels).reshape(-1).astype(np.int64)
        if len(points) != len(labels):
            return None, None

        mask = labels == int(selected_instance)
        if not np.any(mask):
            return None, None

        selected_points = points[mask]
        bounds_min = selected_points.min(axis=0)
        bounds_max = selected_points.max(axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        return center, (bounds_min, bounds_max)

    def log_detected_instances(self, prediction):
        embedding = getattr(prediction, 'embedding', None)
        if embedding is None:
            self.get_logger().warn("ICG-Net prediction has no embedding details for instance logging.")
            return

        point_labels = self.to_numpy(getattr(embedding, 'pointwise_labels', None))
        if point_labels is not None and point_labels.size > 0:
            labels = point_labels.reshape(-1).astype(np.int64)
            instance_ids, counts = np.unique(labels, return_counts=True)
            order = np.argsort(counts)[::-1]
            entries = [
                f"id={int(instance_ids[i])} points={int(counts[i])}"
                for i in order[:10]
            ]
            suffix = "" if len(instance_ids) <= 10 else f"; +{len(instance_ids) - 10} more"
            self.get_logger().info(
                "ICG-Net detected object instances from point labels: "
                f"count={len(instance_ids)} ({'; '.join(entries)}{suffix})"
            )
            return

        class_labels = self.to_numpy(getattr(embedding, 'class_labels', None))
        if class_labels is not None and class_labels.size > 0:
            labels = np.unique(class_labels.reshape(-1).astype(np.int64))
            self.get_logger().info(
                "ICG-Net detected object instance labels: "
                f"{', '.join(str(int(label)) for label in labels)}"
            )
            return

        self.get_logger().warn("ICG-Net did not expose instance labels for this prediction.")

    def log_candidate_instances(self, score_np, width_np, instance_np, valid_mask):
        instance_ids = np.unique(instance_np.astype(np.int64))
        summaries = []
        for instance_id in instance_ids:
            instance_mask = instance_np == instance_id
            candidate_count = int(instance_mask.sum())
            above_count = int((instance_mask & valid_mask).sum())
            local_indices = np.flatnonzero(instance_mask)
            best_index = int(local_indices[np.argmax(score_np[instance_mask])])
            width_mm = width_np[best_index] * 1000.0
            summaries.append(
                (
                    above_count,
                    candidate_count,
                    float(score_np[best_index]),
                    f"id={int(instance_id)} candidates={candidate_count} "
                    f"above={above_count} best_score={score_np[best_index]:.3f} "
                    f"width={width_mm:.1f}mm"
                )
            )

        summaries.sort(key=lambda item: (item[0], item[2], item[1]), reverse=True)
        visible = [entry[-1] for entry in summaries[:8]]
        suffix = "" if len(summaries) <= 8 else f"; +{len(summaries) - 8} more"
        self.get_logger().info(
            f"Grasp candidate summary by object instance: {'; '.join(visible)}{suffix}"
        )

    def log_selected_grasp_plan(
        self,
        grasp_pose_icg,
        grasp_pose_base,
        command_pose_base,
        planner_debug,
        T_ee_to_base,
        T_cam_to_base,
    ):
        target_icg_pos_m = grasp_pose_icg[:3, 3]
        icg_rpy_deg = R.from_matrix(grasp_pose_icg[:3, :3]).as_euler('xyz', degrees=True)
        base_pos_m = grasp_pose_base[:3, 3]
        base_rpy_deg = R.from_matrix(grasp_pose_base[:3, :3]).as_euler('xyz', degrees=True)
        command_pos_mm = command_pose_base[:3, 3] * 1000.0
        command_rpy_deg = R.from_matrix(command_pose_base[:3, :3]).as_euler('xyz', degrees=True)
        ee_world_pos_mm = T_ee_to_base[:3, 3] * 1000.0
        camera_world_pos_mm = T_cam_to_base[:3, 3] * 1000.0

        instance = planner_debug.get('selected_instance')
        score = planner_debug.get('selected_score')
        width_m = planner_debug.get('selected_width_m')
        instance_text = "unknown" if instance is None else str(instance)
        score_text = "unknown" if score is None else f"{score:.3f}"
        width_text = "unknown" if width_m is None or not np.isfinite(width_m) else f"{width_m * 1000.0:.1f}mm"
        contact_camera_m = planner_debug.get('selected_contact_camera_m')
        object_center_camera_m = planner_debug.get('selected_object_center_camera_m')
        geometry_target_base_m = planner_debug.get('selected_geometry_target_base_m')

        self.get_logger().info(
            "Selected grasp plan: "
            f"object_instance={instance_text}, score={score_text}, gripper_width={width_text}"
        )
        self.get_logger().info(
            "Selected target pose in ICG/world frame: "
            f"position_m={self.format_vector(target_icg_pos_m, 4)}, "
            f"rpy_deg={self.format_vector(icg_rpy_deg, 2)}"
        )
        self.get_logger().info(
            "Selected grasp base pose: "
            f"position_m={self.format_vector(base_pos_m, 4)}, "
            f"rpy_deg={self.format_vector(base_rpy_deg, 2)}"
        )
        if contact_camera_m is not None:
            contact_world_m = np.asarray(contact_camera_m, dtype=np.float64)
            self.get_logger().info(
                "Selected object contact point in world [mm]: "
                f"{self.format_vector(contact_world_m * 1000.0, 1)}"
            )
        if object_center_camera_m is not None:
            object_center_world_m = np.asarray(object_center_camera_m, dtype=np.float64)
            self.get_logger().info(
                "Selected object instance center in world [mm]: "
                f"{self.format_vector(object_center_world_m * 1000.0, 1)}"
            )
        if geometry_target_base_m is not None:
            self.get_logger().info(
                "Selected raw geometric object target in world [mm]: "
                f"{self.format_vector(np.asarray(geometry_target_base_m) * 1000.0, 1)}"
            )

        axes = planner_debug.get('selected_gripper_axes_camera')
        if axes is None:
            axes = grasp_pose_icg[:3, :3]
        self.get_logger().info(
            "Selected grasp axes in ICG/world frame: "
            f"x={self.format_vector(axes[:, 0], 3)}, "
            f"y={self.format_vector(axes[:, 1], 3)}, "
            f"z={self.format_vector(axes[:, 2], 3)}"
        )
        self.get_logger().info(
            "World positions [mm]: "
            f"camera={self.format_vector(camera_world_pos_mm, 1)}, "
            f"current_end_effector={self.format_vector(ee_world_pos_mm, 1)}, "
            f"command_payload_target={self.format_vector(command_pos_mm, 1)}"
        )
        self.get_logger().info(
            "Final arm command target: "
            f"position_mm={self.format_vector(command_pos_mm, 1)}, "
            f"rpy_deg={self.format_vector(command_rpy_deg, 1)}, "
            "orientation_source=current_end_effector, "
            f"end_effector={self.gripper_open_value}"
        )

    def select_object_target_in_base(self, planner_debug, T_cam_to_base, grasp_pose_base):
        geometry_target_base_m = planner_debug.get('selected_geometry_target_base_m')
        if geometry_target_base_m is not None:
            target_base_m = np.asarray(geometry_target_base_m, dtype=np.float64)
            component_index = planner_debug.get('selected_geometric_component')
            self.get_logger().info(
                "Payload XYZ source: selected above-table geometric object component "
                f"#{component_index} in world [mm] "
                f"{self.format_vector(target_base_m * 1000.0, 1)}"
            )
            return target_base_m

        object_center_camera_m = planner_debug.get('selected_object_center_camera_m')
        if object_center_camera_m is not None:
            target_base_m = np.asarray(object_center_camera_m, dtype=np.float64)
            if target_base_m[2] > self.geometric_target_min_height:
                self.get_logger().info(
                    "Payload XYZ source: selected ICG object instance center in world [mm] "
                    f"{self.format_vector(target_base_m * 1000.0, 1)}"
                )
                return target_base_m

            self.get_logger().warn(
                "Rejected selected ICG instance center because it is too close to the table plane: "
                f"{self.format_vector(target_base_m * 1000.0, 1)} mm"
            )

        contact_camera_m = planner_debug.get('selected_contact_camera_m')
        if contact_camera_m is not None:
            target_base_m = np.asarray(contact_camera_m, dtype=np.float64)
            self.get_logger().warn(
                "No object instance center available; using selected contact point in world [mm] "
                f"{self.format_vector(target_base_m * 1000.0, 1)}"
            )
            return target_base_m

        target_base_m = grasp_pose_base[:3, 3]
        self.get_logger().warn(
            "No object instance center/contact point available; falling back to ICG-Net gripper pose [mm] "
            f"{self.format_vector(target_base_m * 1000.0, 1)}"
        )
        return target_base_m

    def camera_pose_from_ee_pose(self, T_ee_to_base):
        if self.camera_pose_mode == 'end_effector':
            T_cam_to_base = T_ee_to_base.copy()
            T_cam_to_base[:3, 3] += self.camera_offset_from_ee_m
            return T_cam_to_base

        T_cam_to_base = np.eye(4)
        T_cam_to_base[:3, :3] = R.from_euler(
            'xyz',
            self.fixed_camera_rpy_deg,
            degrees=True,
        ).as_matrix()
        T_cam_to_base[:3, 3] = self.fixed_camera_position_m
        return T_cam_to_base

    def cart_pos_to_matrix(self, cart_pos):
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = cart_pos
        transform = np.eye(4)
        transform[:3, :3] = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
        transform[:3, 3] = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
        return transform

    def publish_arm_commands(self, x, y, z, roll, pitch, yaw, gripper_val):
        # 1. Publish arm mode set to [1, 1, 0, 0]
        mode_msg = Int32MultiArray()
        mode_msg.data = [1, 1, 0, 0]
        self.mode_pub.publish(mode_msg)
        self.get_logger().info(f"Published arm mode: {mode_msg.data}")

        # 2. Publish payload set: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg, end_effector]
        payload_msg = Int32MultiArray()
        payload_msg.data = [x, y, z, roll, pitch, yaw, gripper_val]
        self.payload_pub.publish(payload_msg)
        self.get_logger().info(f"Published payload command: {payload_msg.data}")

    def publish_grasp_pose_vis(self, grasp_pose_base):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.base_frame

        # Extract translation
        pose_msg.pose.position.x = grasp_pose_base[0, 3]
        pose_msg.pose.position.y = grasp_pose_base[1, 3]
        pose_msg.pose.position.z = grasp_pose_base[2, 3]

        # Extract rotation quaternion
        q = R.from_matrix(grasp_pose_base[:3, :3]).as_quat()
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
