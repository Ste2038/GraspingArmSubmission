# ROS 2 Grasping Wrapper

`ros2_grasping` connects ICG-Net grasp planning to the robot-arm ROS 2 API. It
receives a camera point cloud and the current Cartesian arm pose, filters the
table workspace, runs ICG-Net, selects an object target, and publishes the arm
command.

## Build and Run

Fetch the official repositories and checkpoint first as described in the root
README. From the repository root:

```bash
colcon build --packages-select ros2_grasping
source install/setup.bash
ros2 run ros2_grasping grasp_planner_node
```

The default model files are resolved relative to the repository:

```text
third_party/icg_benchmark/data/icgnet/51--0.656/config.yaml
third_party/icg_benchmark/data/icgnet/51--0.656/checkpoint.ckpt
```

Override parameters with standard ROS arguments:

```bash
ros2 run ros2_grasping grasp_planner_node --ros-args \
  -p camera_pose_mode:=end_effector \
  -p confidence_threshold:=0.45
```

## ROS API

### Subscriptions

| Topic | Type | Content |
| --- | --- | --- |
| `/FRT/SW/track/zed/hand/pointcloud` | `sensor_msgs/msg/PointCloud2` | Point cloud in the camera frame |
| `/FRT/arm/L0/cartPos` | `std_msgs/msg/Float32MultiArray` | `[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]` |

The Cartesian-pose subscription uses best-effort QoS to match the arm
publisher.

### Publishers

| Topic | Type | Content |
| --- | --- | --- |
| `/FRT/arm/L0/mode/set` | `std_msgs/msg/Int32MultiArray` | Arm mode `[1, 1, 0, 0]` |
| `/FRT/arm/L0/payload/set` | `std_msgs/msg/Int32MultiArray` | Target pose and gripper command |
| `/grasp_pose` | `geometry_msgs/msg/PoseStamped` | Selected command pose in `base_frame` |

The payload is:

```text
[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg, end_effector]
```

### Services

| Service | Type | Action |
| --- | --- | --- |
| `/plan_grasp` | `std_srvs/srv/Trigger` | Plan a grasp and publish the target |
| `/open_gripper` | `std_srvs/srv/Trigger` | Publish the open-gripper command |
| `/close_gripper` | `std_srvs/srv/Trigger` | Publish the close-gripper command |

Example:

```bash
ros2 service call /plan_grasp std_srvs/srv/Trigger '{}'
```

## Camera Calibration

`camera_pose_mode` selects the transform used for the incoming point cloud:

- `fixed` uses the calibrated world pose and is the default for simulation.
- `end_effector` uses the current end-effector orientation and adds the
  configured XYZ offset to its world position.

Default fixed pose:

```text
position: [0.490, 0.586, 0.489] m
RPY:      [1.0, 34.7, -99.5] degrees
```

Default end-effector offset:

```text
[-0.160, 0.060, 0.060] m
```

The point-cloud origin correction defaults to `[0.060, 0.0, 0.0]` m in the
camera frame. The default base-frame workspace crop is:

```text
minimum: [0.2, -0.5, -0.01] m
maximum: [0.9,  0.5,  0.20] m
```

ICG-Net receives the cropped cloud in the base/world frame so its table-height
and grasp filters operate along world `+Z`.

## Parameters

### Model and Planning

| Parameter | Default | Description |
| --- | --- | --- |
| `config_path` | repository model config | ICG-Net configuration |
| `checkpoint_path` | repository checkpoint | ICG-Net weights |
| `device` | `cuda` | Inference device; falls back to CPU when needed |
| `confidence_threshold` | `0.4` | Minimum grasp confidence |
| `max_gripper_width` | `0.08` | Maximum width in metres |
| `voxel_size` | `0.0045` | Point-cloud downsampling size |
| `base_frame` | `base_link` | Output pose frame |
| `geometric_target_object_index` | `0` | Ranked geometric component to target |
| `geometric_target_min_height` | `0.025` | Minimum component height in metres |
| `geometric_target_max_height` | `0.18` | Maximum component height in metres |

### Camera and Workspace

| Parameter group | Defaults |
| --- | --- |
| `camera_pose_mode` | `fixed` |
| `fixed_camera_{x,y,z}` | `0.490, 0.586, 0.489` m |
| `fixed_camera_{roll,pitch,yaw}` | `1.0, 34.7, -99.5` degrees |
| `camera_offset_{x,y,z}` | `-0.160, 0.060, 0.060` m |
| `pointcloud_origin_offset_{x,y,z}` | `0.060, 0.0, 0.0` m |
| `table_crop_min_{x,y,z}` | `0.2, -0.5, -0.01` m |
| `table_crop_max_{x,y,z}` | `0.9, 0.5, 0.2` m |

### Arm and Gripper

| Parameter | Default | Description |
| --- | --- | --- |
| `command_tcp_directly` | `true` | Treat selected pose as the TCP command |
| `tcp_offset_{x,y,z}` | `-0.01, 0.0, 0.121` m | TCP offset from the hand flange |
| `gripper_open_value` | `100` | Open command value |
| `gripper_close_value` | `50` | Close command value |

The selected object center is preferred as the command target, followed by the
selected contact point and grasp pose. The current end-effector orientation is
preserved for the arm payload.

## Point-Cloud Capture

Optional capture is disabled by default. Enable it without editing source:

```bash
ros2 run ros2_grasping grasp_planner_node --ros-args \
  -p save_debug_pointclouds:=true
```

| Parameter | Default | Description |
| --- | --- | --- |
| `save_debug_pointclouds` | `false` | Capture incoming cropped clouds |
| `debug_pointcloud_dir` | `debug/pointclouds` | Repository-relative output directory |
| `save_debug_pointcloud_npz` | `true` | Write NumPy data and transforms |
| `save_debug_pointcloud_ply` | `true` | Write Open3D-compatible point clouds |

Each capture includes timestamped output, a `latest_pointcloud` copy, and JSON
calibration metadata. The `debug/` directory is ignored by git.
