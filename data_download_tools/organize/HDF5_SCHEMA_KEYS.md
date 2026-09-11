# HDF5 Schema完整Key清单

- 来源：`/media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/schema_state/schema_catalog.json`
- Logical schema数量：11
- Storage schema数量：11
- Shape中的 `T` 表示与 `metadata/trajectory_length` 对齐的时间长度。
- Shape中的 `N` 表示非对齐原始采样长度。
- 本次11种schema的Group和Dataset均未记录额外attribute key。

## schema_25da69addf78

- 文件数：2319
- Group数：45
- Dataset数：122
- 工站：tienyi_31, tienyi_7, tienyi_9
- 任务：back_battery_and_cover, insert_hose, plug_cables
- Arm mode：dual
- Puppet pose：True
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_head, camera_left, camera_right
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_31/back_battery_and_cover/tienyi_prod2_dualArm-gripper-3cameras_515_Back_small_battery_installation_and_small_battery_cover_installation_20260821/success_episodes/0821_145350/data/trajectory.hdf5`

### Group keys（45）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
force_observations
force_observations/force_left_end_effector_align
force_observations/force_left_end_effector_raw
force_observations/force_right_end_effector_align
force_observations/force_right_end_effector_raw
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_pose_align
puppet/end_effector_left_pose_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_pose_align
puppet/end_effector_right_pose_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_pose_align
puppet/head_pose_raw
puppet/head_position_align
puppet/head_position_raw
```

### Dataset keys（122）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_head` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_head` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_head` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_left_arm_left` | `[4, 4]` | `float64` |
| 12 | `camera_extrinsics/camera_right_arm_right` | `[4, 4]` | `float64` |
| 13 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 14 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 15 | `camera_model/camera_head` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 17 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 18 | `camera_observations/color_images/camera_head` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_head` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 24 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 25 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 26 | `force_observations/force_left_end_effector_align/data` | `["T", 6]` | `float64` |
| 27 | `force_observations/force_left_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 28 | `force_observations/force_left_end_effector_align/timestamp` | `["T"]` | `float64` |
| 29 | `force_observations/force_left_end_effector_raw/data` | `["N", 6]` | `float64` |
| 30 | `force_observations/force_left_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 31 | `force_observations/force_left_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 32 | `force_observations/force_right_end_effector_align/data` | `["T", 6]` | `float64` |
| 33 | `force_observations/force_right_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 34 | `force_observations/force_right_end_effector_align/timestamp` | `["T"]` | `float64` |
| 35 | `force_observations/force_right_end_effector_raw/data` | `["N", 6]` | `float64` |
| 36 | `force_observations/force_right_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 37 | `force_observations/force_right_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 38 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 39 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 40 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 41 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 42 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 43 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 44 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 45 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 46 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 47 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 48 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 49 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 50 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 51 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 52 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 53 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 54 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 55 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 56 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 57 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 58 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 59 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 60 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 61 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 62 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 63 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 64 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 65 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 66 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 67 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 68 | `metadata/collection_time` | `[]` | `vlen_string` |
| 69 | `metadata/collector` | `[]` | `vlen_string` |
| 70 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 71 | `metadata/data_type` | `[]` | `vlen_string` |
| 72 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 73 | `metadata/trajectory_length` | `[]` | `int64` |
| 74 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 75 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 76 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 77 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 78 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 79 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 80 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 81 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 82 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 83 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 84 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 85 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 86 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 87 | `puppet/end_effector_left_pose_align/data` | `["T", 7]` | `float32` |
| 88 | `puppet/end_effector_left_pose_align/is_intervene` | `["T"]` | `bool` |
| 89 | `puppet/end_effector_left_pose_align/timestamp` | `["T"]` | `float64` |
| 90 | `puppet/end_effector_left_pose_raw/data` | `["N", 7]` | `float32` |
| 91 | `puppet/end_effector_left_pose_raw/is_intervene` | `["N"]` | `bool` |
| 92 | `puppet/end_effector_left_pose_raw/timestamp` | `["N"]` | `float64` |
| 93 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 94 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 95 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 96 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 97 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 98 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 99 | `puppet/end_effector_right_pose_align/data` | `["T", 7]` | `float32` |
| 100 | `puppet/end_effector_right_pose_align/is_intervene` | `["T"]` | `bool` |
| 101 | `puppet/end_effector_right_pose_align/timestamp` | `["T"]` | `float64` |
| 102 | `puppet/end_effector_right_pose_raw/data` | `["N", 7]` | `float32` |
| 103 | `puppet/end_effector_right_pose_raw/is_intervene` | `["N"]` | `bool` |
| 104 | `puppet/end_effector_right_pose_raw/timestamp` | `["N"]` | `float64` |
| 105 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 106 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 107 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 108 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 109 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 110 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 111 | `puppet/head_pose_align/data` | `["T", 7]` | `float32` |
| 112 | `puppet/head_pose_align/is_intervene` | `["T"]` | `bool` |
| 113 | `puppet/head_pose_align/timestamp` | `["T"]` | `float64` |
| 114 | `puppet/head_pose_raw/data` | `["N", 7]` | `float32` |
| 115 | `puppet/head_pose_raw/is_intervene` | `["N"]` | `bool` |
| 116 | `puppet/head_pose_raw/timestamp` | `["N"]` | `float64` |
| 117 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 118 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 119 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 120 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 121 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 122 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |

## schema_57394ccaae75

- 文件数：1731
- Group数：32
- Dataset数：86
- 工站：tienyi_30, tienyi_7, tienyi_9
- 任务：back_handle, insert_hose, plug_cables
- Arm mode：dual
- Puppet pose：False
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_left, camera_right, camera_top
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_30/back_handle/tienyi_prod2_dualArm-gripper-3cameras_368_back-handle-installation_20260811/success_episodes/0811_141918/data/trajectory.hdf5`

### Group keys（32）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
```

### Dataset keys（86）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_top` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_top` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_top` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_left_arm_left` | `[4, 4]` | `float64` |
| 12 | `camera_extrinsics/camera_right_arm_right` | `[4, 4]` | `float64` |
| 13 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 14 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 15 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 17 | `camera_model/camera_top` | `[]` | `vlen_string` |
| 18 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/color_images/camera_top` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/depth_images/camera_top` | `["T"]` | `vlen_uint8` |
| 24 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 25 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 26 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 27 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 28 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 29 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 30 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 31 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 32 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 33 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 34 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 35 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 36 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 37 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 38 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 39 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 40 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 41 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 42 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 43 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 44 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 45 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 46 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 47 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 48 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 49 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 50 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 51 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 52 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 53 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 54 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 55 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 56 | `metadata/collection_time` | `[]` | `vlen_string` |
| 57 | `metadata/collector` | `[]` | `vlen_string` |
| 58 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 59 | `metadata/data_type` | `[]` | `vlen_string` |
| 60 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 61 | `metadata/trajectory_length` | `[]` | `int64` |
| 62 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 63 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 64 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 65 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 66 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 67 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 68 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 69 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 70 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 71 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 72 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 73 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 74 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 75 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 76 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 77 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 78 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 79 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 80 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 81 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 82 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 83 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 84 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 85 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 86 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |

## schema_8eb11f9b33ab

- 文件数：1579
- Group数：34
- Dataset数：92
- 工站：tienyi_30, tienyi_7, tienyi_9
- 任务：back_handle, insert_hose, plug_cables
- Arm mode：dual
- Puppet pose：False
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_left, camera_right, camera_top
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_30/back_handle/tienyi_prod2_dualArm-gripper-3cameras_368_back-handle-installation_20260820/success_episodes/0820_084420/data/trajectory.hdf5`

### Group keys（34）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_position_align
puppet/head_position_raw
```

### Dataset keys（92）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_top` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_top` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_top` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_left_arm_left` | `[4, 4]` | `float64` |
| 12 | `camera_extrinsics/camera_right_arm_right` | `[4, 4]` | `float64` |
| 13 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 14 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 15 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 17 | `camera_model/camera_top` | `[]` | `vlen_string` |
| 18 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/color_images/camera_top` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/depth_images/camera_top` | `["T"]` | `vlen_uint8` |
| 24 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 25 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 26 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 27 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 28 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 29 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 30 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 31 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 32 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 33 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 34 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 35 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 36 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 37 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 38 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 39 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 40 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 41 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 42 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 43 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 44 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 45 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 46 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 47 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 48 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 49 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 50 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 51 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 52 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 53 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 54 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 55 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 56 | `metadata/collection_time` | `[]` | `vlen_string` |
| 57 | `metadata/collector` | `[]` | `vlen_string` |
| 58 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 59 | `metadata/data_type` | `[]` | `vlen_string` |
| 60 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 61 | `metadata/trajectory_length` | `[]` | `int64` |
| 62 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 63 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 64 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 65 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 66 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 67 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 68 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 69 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 70 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 71 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 72 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 73 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 74 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 75 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 76 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 77 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 78 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 79 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 80 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 81 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 82 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 83 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 84 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 85 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 86 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 87 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 88 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 89 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 90 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 91 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 92 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |

## schema_9a87c6945e7f

- 文件数：1248
- Group数：31
- Dataset数：81
- 工站：tienyi_9
- 任务：plug_cables
- Arm mode：dual
- Puppet pose：False
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_left, camera_right, camera_top
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_9/plug_cables/tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260810_pm/success_episodes/0810_212251/data/trajectory.hdf5`

### Group keys（31）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
```

### Dataset keys（81）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_top` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_top` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_top` | `[2]` | `int64` |
| 10 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 11 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 12 | `camera_model/camera_top` | `[]` | `vlen_string` |
| 13 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 14 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 15 | `camera_observations/color_images/camera_top` | `["T"]` | `vlen_uint8` |
| 16 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 17 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 18 | `camera_observations/depth_images/camera_top` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 20 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 21 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 22 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 23 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 24 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 25 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 26 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 27 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 28 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 29 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 30 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 31 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 32 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 33 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 34 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 35 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 36 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 37 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 38 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 39 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 40 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 41 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 42 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 43 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 44 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 45 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 46 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 47 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 48 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 49 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 50 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 51 | `metadata/collection_time` | `[]` | `vlen_string` |
| 52 | `metadata/collector` | `[]` | `vlen_string` |
| 53 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 54 | `metadata/data_type` | `[]` | `vlen_string` |
| 55 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 56 | `metadata/trajectory_length` | `[]` | `int64` |
| 57 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 58 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 59 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 60 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 61 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 62 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 63 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 64 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 65 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 66 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 67 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 68 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 69 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 70 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 71 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 72 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 73 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 74 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 75 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 76 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 77 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 78 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 79 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 80 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 81 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |

## schema_ddf65632e20f

- 文件数：424
- Group数：45
- Dataset数：120
- 工站：tienyi_31
- 任务：back_handle
- Arm mode：dual
- Puppet pose：True
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_head, camera_left, camera_right
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_31/back_handle/tienyi_prod2_dualArm-gripper-3cameras_515_back-handle-installation_20260805_pm/success_episodes/0805_194209/data/trajectory.hdf5`

### Group keys（45）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
force_observations
force_observations/force_left_end_effector_align
force_observations/force_left_end_effector_raw
force_observations/force_right_end_effector_align
force_observations/force_right_end_effector_raw
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_pose_align
puppet/end_effector_left_pose_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_pose_align
puppet/end_effector_right_pose_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_pose_align
puppet/head_pose_raw
puppet/head_position_align
puppet/head_position_raw
puppet/waist_position_align
puppet/waist_position_raw
```

### Dataset keys（120）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_head` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_head` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_head` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_left` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 12 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 13 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 14 | `camera_model/camera_head` | `[]` | `vlen_string` |
| 15 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 17 | `camera_observations/color_images/camera_head` | `["T"]` | `vlen_uint8` |
| 18 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/depth_images/camera_head` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 24 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 25 | `force_observations/force_left_end_effector_align/data` | `["T", 6]` | `float64` |
| 26 | `force_observations/force_left_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 27 | `force_observations/force_left_end_effector_align/timestamp` | `["T"]` | `float64` |
| 28 | `force_observations/force_left_end_effector_raw/data` | `["N", 6]` | `float64` |
| 29 | `force_observations/force_left_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 30 | `force_observations/force_left_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 31 | `force_observations/force_right_end_effector_align/data` | `["T", 6]` | `float64` |
| 32 | `force_observations/force_right_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 33 | `force_observations/force_right_end_effector_align/timestamp` | `["T"]` | `float64` |
| 34 | `force_observations/force_right_end_effector_raw/data` | `["N", 6]` | `float64` |
| 35 | `force_observations/force_right_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 36 | `force_observations/force_right_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 37 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 38 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 39 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 40 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 41 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 42 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 43 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 44 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 45 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 46 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 47 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 48 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 49 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 50 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 51 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 52 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 53 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 54 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 55 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 56 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 57 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 58 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 59 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 60 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 61 | `metadata/collection_time` | `[]` | `vlen_string` |
| 62 | `metadata/collector` | `[]` | `vlen_string` |
| 63 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 64 | `metadata/data_type` | `[]` | `vlen_string` |
| 65 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 66 | `metadata/trajectory_length` | `[]` | `int64` |
| 67 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 68 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 69 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 70 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 71 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 72 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 73 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 74 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 75 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 76 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 77 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 78 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 79 | `puppet/end_effector_left_pose_align/data` | `["T", 7]` | `float32` |
| 80 | `puppet/end_effector_left_pose_align/is_intervene` | `["T"]` | `bool` |
| 81 | `puppet/end_effector_left_pose_align/timestamp` | `["T"]` | `float64` |
| 82 | `puppet/end_effector_left_pose_raw/data` | `["N", 7]` | `float32` |
| 83 | `puppet/end_effector_left_pose_raw/is_intervene` | `["N"]` | `bool` |
| 84 | `puppet/end_effector_left_pose_raw/timestamp` | `["N"]` | `float64` |
| 85 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 86 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 87 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 88 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 89 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 90 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 91 | `puppet/end_effector_right_pose_align/data` | `["T", 7]` | `float32` |
| 92 | `puppet/end_effector_right_pose_align/is_intervene` | `["T"]` | `bool` |
| 93 | `puppet/end_effector_right_pose_align/timestamp` | `["T"]` | `float64` |
| 94 | `puppet/end_effector_right_pose_raw/data` | `["N", 7]` | `float32` |
| 95 | `puppet/end_effector_right_pose_raw/is_intervene` | `["N"]` | `bool` |
| 96 | `puppet/end_effector_right_pose_raw/timestamp` | `["N"]` | `float64` |
| 97 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 98 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 99 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 100 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 101 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 102 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 103 | `puppet/head_pose_align/data` | `["T", 7]` | `float32` |
| 104 | `puppet/head_pose_align/is_intervene` | `["T"]` | `bool` |
| 105 | `puppet/head_pose_align/timestamp` | `["T"]` | `float64` |
| 106 | `puppet/head_pose_raw/data` | `["N", 7]` | `float32` |
| 107 | `puppet/head_pose_raw/is_intervene` | `["N"]` | `bool` |
| 108 | `puppet/head_pose_raw/timestamp` | `["N"]` | `float64` |
| 109 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 110 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 111 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 112 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 113 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 114 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |
| 115 | `puppet/waist_position_align/data` | `["T", 2]` | `float32` |
| 116 | `puppet/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 117 | `puppet/waist_position_align/timestamp` | `["T"]` | `float64` |
| 118 | `puppet/waist_position_raw/data` | `["N", 2]` | `float32` |
| 119 | `puppet/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 120 | `puppet/waist_position_raw/timestamp` | `["N"]` | `float64` |

## schema_fbcd96301bc5

- 文件数：239
- Group数：34
- Dataset数：92
- 工站：tienyi_30
- 任务：back_handle
- Arm mode：dual
- Puppet pose：False
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_head, camera_left, camera_right
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_30/back_handle/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260821/success_episodes/0822_092840/data/trajectory.hdf5`

### Group keys（34）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_position_align
puppet/head_position_raw
```

### Dataset keys（92）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_head` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_head` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_head` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_left_arm_left` | `[4, 4]` | `float64` |
| 12 | `camera_extrinsics/camera_right_arm_right` | `[4, 4]` | `float64` |
| 13 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 14 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 15 | `camera_model/camera_head` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 17 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 18 | `camera_observations/color_images/camera_head` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_head` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 24 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 25 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 26 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 27 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 28 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 29 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 30 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 31 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 32 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 33 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 34 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 35 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 36 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 37 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 38 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 39 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 40 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 41 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 42 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 43 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 44 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 45 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 46 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 47 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 48 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 49 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 50 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 51 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 52 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 53 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 54 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 55 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 56 | `metadata/collection_time` | `[]` | `vlen_string` |
| 57 | `metadata/collector` | `[]` | `vlen_string` |
| 58 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 59 | `metadata/data_type` | `[]` | `vlen_string` |
| 60 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 61 | `metadata/trajectory_length` | `[]` | `int64` |
| 62 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 63 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 64 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 65 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 66 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 67 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 68 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 69 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 70 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 71 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 72 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 73 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 74 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 75 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 76 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 77 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 78 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 79 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 80 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 81 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 82 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 83 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 84 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 85 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 86 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 87 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 88 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 89 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 90 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 91 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 92 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |

## schema_37d21d6c63eb

- 文件数：236
- Group数：45
- Dataset数：121
- 工站：tienyi_31
- 任务：back_battery_and_cover
- Arm mode：dual
- Puppet pose：True
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_head, camera_left, camera_right
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_31/back_battery_and_cover/tienyi_prod2_dualArm-gripper-3cameras_515_Back_small_battery_installation_and_small_battery_cover_installation_20260824/success_episodes/0824_135113/data/trajectory.hdf5`

### Group keys（45）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
force_observations
force_observations/force_left_end_effector_align
force_observations/force_left_end_effector_raw
force_observations/force_right_end_effector_align
force_observations/force_right_end_effector_raw
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_pose_align
puppet/end_effector_left_pose_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_pose_align
puppet/end_effector_right_pose_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_pose_align
puppet/head_pose_raw
puppet/head_position_align
puppet/head_position_raw
```

### Dataset keys（121）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_head` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_head` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_head` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_left_arm_left` | `[4, 4]` | `float64` |
| 12 | `camera_extrinsics/camera_right_arm_right` | `[4, 4]` | `float64` |
| 13 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 14 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 15 | `camera_model/camera_head` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 17 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 18 | `camera_observations/color_images/camera_head` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_head` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 24 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 25 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 26 | `force_observations/force_left_end_effector_align/data` | `["T", 6]` | `float64` |
| 27 | `force_observations/force_left_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 28 | `force_observations/force_left_end_effector_align/timestamp` | `["T"]` | `float64` |
| 29 | `force_observations/force_left_end_effector_raw/data` | `["N", 6]` | `float64` |
| 30 | `force_observations/force_left_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 31 | `force_observations/force_left_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 32 | `force_observations/force_right_end_effector_align/data` | `["T", 6]` | `float64` |
| 33 | `force_observations/force_right_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 34 | `force_observations/force_right_end_effector_align/timestamp` | `["T"]` | `float64` |
| 35 | `force_observations/force_right_end_effector_raw/data` | `["N", 6]` | `float64` |
| 36 | `force_observations/force_right_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 37 | `force_observations/force_right_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 38 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 39 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 40 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 41 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 42 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 43 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 44 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 45 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 46 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 47 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 48 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 49 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 50 | `master/end_effector_left_position_align/data` | `["T"]` | `float64` |
| 51 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 52 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 53 | `master/end_effector_left_position_raw/data` | `["N"]` | `float64` |
| 54 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 55 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 56 | `master/end_effector_right_position_align/data` | `["T"]` | `float64` |
| 57 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 58 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 59 | `master/end_effector_right_position_raw/data` | `["N"]` | `float64` |
| 60 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 61 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 62 | `master/waist_position_align/data` | `["T"]` | `float64` |
| 63 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 64 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 65 | `master/waist_position_raw/data` | `["N"]` | `float64` |
| 66 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 67 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 68 | `metadata/collection_time` | `[]` | `vlen_string` |
| 69 | `metadata/collector` | `[]` | `vlen_string` |
| 70 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 71 | `metadata/data_type` | `[]` | `vlen_string` |
| 72 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 73 | `metadata/trajectory_length` | `[]` | `int64` |
| 74 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 75 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 76 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 77 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 78 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 79 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 80 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 81 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 82 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 83 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 84 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 85 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 86 | `puppet/end_effector_left_pose_align/data` | `["T", 7]` | `float32` |
| 87 | `puppet/end_effector_left_pose_align/is_intervene` | `["T"]` | `bool` |
| 88 | `puppet/end_effector_left_pose_align/timestamp` | `["T"]` | `float64` |
| 89 | `puppet/end_effector_left_pose_raw/data` | `["N", 7]` | `float32` |
| 90 | `puppet/end_effector_left_pose_raw/is_intervene` | `["N"]` | `bool` |
| 91 | `puppet/end_effector_left_pose_raw/timestamp` | `["N"]` | `float64` |
| 92 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 93 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 94 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 95 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 96 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 97 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 98 | `puppet/end_effector_right_pose_align/data` | `["T", 7]` | `float32` |
| 99 | `puppet/end_effector_right_pose_align/is_intervene` | `["T"]` | `bool` |
| 100 | `puppet/end_effector_right_pose_align/timestamp` | `["T"]` | `float64` |
| 101 | `puppet/end_effector_right_pose_raw/data` | `["N", 7]` | `float32` |
| 102 | `puppet/end_effector_right_pose_raw/is_intervene` | `["N"]` | `bool` |
| 103 | `puppet/end_effector_right_pose_raw/timestamp` | `["N"]` | `float64` |
| 104 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 105 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 106 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 107 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 108 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 109 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 110 | `puppet/head_pose_align/data` | `["T", 7]` | `float32` |
| 111 | `puppet/head_pose_align/is_intervene` | `["T"]` | `bool` |
| 112 | `puppet/head_pose_align/timestamp` | `["T"]` | `float64` |
| 113 | `puppet/head_pose_raw/data` | `["N", 7]` | `float32` |
| 114 | `puppet/head_pose_raw/is_intervene` | `["N"]` | `bool` |
| 115 | `puppet/head_pose_raw/timestamp` | `["N"]` | `float64` |
| 116 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 117 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 118 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 119 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 120 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 121 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |

## schema_075c3cd43453

- 文件数：167
- Group数：40
- Dataset数：109
- 工站：unknown_station
- 任务：back_handle
- Arm mode：dual
- Puppet pose：True
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_head, camera_left, camera_right
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/unknown_station/back_handle/back_handle_20260803_pm_failure/success_episodes/0803_172504/data/trajectory.hdf5`

### Group keys（40）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_pose_align
puppet/end_effector_left_pose_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_pose_align
puppet/end_effector_right_pose_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_pose_align
puppet/head_pose_raw
puppet/head_position_align
puppet/head_position_raw
puppet/waist_position_align
puppet/waist_position_raw
```

### Dataset keys（109）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_head` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_head` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_head` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_left` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 12 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 13 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 14 | `camera_model/camera_head` | `[]` | `vlen_string` |
| 15 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 17 | `camera_observations/color_images/camera_head` | `["T"]` | `vlen_uint8` |
| 18 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/depth_images/camera_head` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 24 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 25 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 26 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 27 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 28 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 29 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 30 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 31 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 32 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 33 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 34 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 35 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 36 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 37 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 38 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 39 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 40 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 41 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 42 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 43 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 44 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 45 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 46 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 47 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 48 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 49 | `metadata/collection_time` | `[]` | `vlen_string` |
| 50 | `metadata/collector` | `[]` | `vlen_string` |
| 51 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 52 | `metadata/data_type` | `[]` | `vlen_string` |
| 53 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 54 | `metadata/trajectory_length` | `[]` | `int64` |
| 55 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 56 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 57 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 58 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 59 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 60 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 61 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 62 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 63 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 64 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 65 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 66 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 67 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 68 | `puppet/end_effector_left_pose_align/data` | `["T", 7]` | `float32` |
| 69 | `puppet/end_effector_left_pose_align/is_intervene` | `["T"]` | `bool` |
| 70 | `puppet/end_effector_left_pose_align/timestamp` | `["T"]` | `float64` |
| 71 | `puppet/end_effector_left_pose_raw/data` | `["N", 7]` | `float32` |
| 72 | `puppet/end_effector_left_pose_raw/is_intervene` | `["N"]` | `bool` |
| 73 | `puppet/end_effector_left_pose_raw/timestamp` | `["N"]` | `float64` |
| 74 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 75 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 76 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 77 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 78 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 79 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 80 | `puppet/end_effector_right_pose_align/data` | `["T", 7]` | `float32` |
| 81 | `puppet/end_effector_right_pose_align/is_intervene` | `["T"]` | `bool` |
| 82 | `puppet/end_effector_right_pose_align/timestamp` | `["T"]` | `float64` |
| 83 | `puppet/end_effector_right_pose_raw/data` | `["N", 7]` | `float32` |
| 84 | `puppet/end_effector_right_pose_raw/is_intervene` | `["N"]` | `bool` |
| 85 | `puppet/end_effector_right_pose_raw/timestamp` | `["N"]` | `float64` |
| 86 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 87 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 88 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 89 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 90 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 91 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 92 | `puppet/head_pose_align/data` | `["T", 7]` | `float32` |
| 93 | `puppet/head_pose_align/is_intervene` | `["T"]` | `bool` |
| 94 | `puppet/head_pose_align/timestamp` | `["T"]` | `float64` |
| 95 | `puppet/head_pose_raw/data` | `["N", 7]` | `float32` |
| 96 | `puppet/head_pose_raw/is_intervene` | `["N"]` | `bool` |
| 97 | `puppet/head_pose_raw/timestamp` | `["N"]` | `float64` |
| 98 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 99 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 100 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 101 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 102 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 103 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |
| 104 | `puppet/waist_position_align/data` | `["T", 2]` | `float32` |
| 105 | `puppet/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 106 | `puppet/waist_position_align/timestamp` | `["T"]` | `float64` |
| 107 | `puppet/waist_position_raw/data` | `["N", 2]` | `float32` |
| 108 | `puppet/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 109 | `puppet/waist_position_raw/timestamp` | `["N"]` | `float64` |

## schema_4bdb64a581d1

- 文件数：2
- Group数：44
- Dataset数：115
- 工站：tienyi_9
- 任务：plug_cables
- Arm mode：dual
- Puppet pose：True
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：(无实际图像Dataset)
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_9/plug_cables/tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260825_left/success_episodes/0825_151413/data/trajectory.hdf5`

### Group keys（44）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
force_observations
force_observations/force_left_end_effector_align
force_observations/force_left_end_effector_raw
force_observations/force_right_end_effector_align
force_observations/force_right_end_effector_raw
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_pose_align
puppet/end_effector_left_pose_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_pose_align
puppet/end_effector_right_pose_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_pose_align
puppet/head_pose_raw
puppet/head_position_align
puppet/head_position_raw
```

### Dataset keys（115）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_head` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_head` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_head` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_left_arm_left` | `[4, 4]` | `float64` |
| 12 | `camera_extrinsics/camera_right_arm_right` | `[4, 4]` | `float64` |
| 13 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 14 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 15 | `camera_model/camera_head` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 17 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 18 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 19 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 20 | `force_observations/force_left_end_effector_align/data` | `["T", 6]` | `float64` |
| 21 | `force_observations/force_left_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 22 | `force_observations/force_left_end_effector_align/timestamp` | `["T"]` | `float64` |
| 23 | `force_observations/force_left_end_effector_raw/data` | `["N", 6]` | `float64` |
| 24 | `force_observations/force_left_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 25 | `force_observations/force_left_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 26 | `force_observations/force_right_end_effector_align/data` | `["T", 6]` | `float64` |
| 27 | `force_observations/force_right_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 28 | `force_observations/force_right_end_effector_align/timestamp` | `["T"]` | `float64` |
| 29 | `force_observations/force_right_end_effector_raw/data` | `["N", 6]` | `float64` |
| 30 | `force_observations/force_right_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 31 | `force_observations/force_right_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 32 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 33 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 34 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 35 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 36 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 37 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 38 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 39 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 40 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 41 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 42 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 43 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 44 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 45 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 46 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 47 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 48 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 49 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 50 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 51 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 52 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 53 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 54 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 55 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 56 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 57 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 58 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 59 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 60 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 61 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 62 | `metadata/collection_time` | `[]` | `vlen_string` |
| 63 | `metadata/collector` | `[]` | `vlen_string` |
| 64 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 65 | `metadata/data_type` | `[]` | `vlen_string` |
| 66 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 67 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 68 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 69 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 70 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 71 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 72 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 73 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 74 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 75 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 76 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 77 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 78 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 79 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 80 | `puppet/end_effector_left_pose_align/data` | `["T", 7]` | `float32` |
| 81 | `puppet/end_effector_left_pose_align/is_intervene` | `["T"]` | `bool` |
| 82 | `puppet/end_effector_left_pose_align/timestamp` | `["T"]` | `float64` |
| 83 | `puppet/end_effector_left_pose_raw/data` | `["N", 7]` | `float32` |
| 84 | `puppet/end_effector_left_pose_raw/is_intervene` | `["N"]` | `bool` |
| 85 | `puppet/end_effector_left_pose_raw/timestamp` | `["N"]` | `float64` |
| 86 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 87 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 88 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 89 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 90 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 91 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 92 | `puppet/end_effector_right_pose_align/data` | `["T", 7]` | `float32` |
| 93 | `puppet/end_effector_right_pose_align/is_intervene` | `["T"]` | `bool` |
| 94 | `puppet/end_effector_right_pose_align/timestamp` | `["T"]` | `float64` |
| 95 | `puppet/end_effector_right_pose_raw/data` | `["N", 7]` | `float32` |
| 96 | `puppet/end_effector_right_pose_raw/is_intervene` | `["N"]` | `bool` |
| 97 | `puppet/end_effector_right_pose_raw/timestamp` | `["N"]` | `float64` |
| 98 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 99 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 100 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 101 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 102 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 103 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 104 | `puppet/head_pose_align/data` | `["T", 7]` | `float32` |
| 105 | `puppet/head_pose_align/is_intervene` | `["T"]` | `bool` |
| 106 | `puppet/head_pose_align/timestamp` | `["T"]` | `float64` |
| 107 | `puppet/head_pose_raw/data` | `["N", 7]` | `float32` |
| 108 | `puppet/head_pose_raw/is_intervene` | `["N"]` | `bool` |
| 109 | `puppet/head_pose_raw/timestamp` | `["N"]` | `float64` |
| 110 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 111 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 112 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 113 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 114 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 115 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |

## schema_3cbf90453c18

- 文件数：1
- Group数：45
- Dataset数：121
- 工站：tienyi_9
- 任务：plug_cables
- Arm mode：dual
- Puppet pose：True
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：camera_head, camera_left, camera_right
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_9/plug_cables/tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260824_left/success_episodes/0824_151329/data/trajectory.hdf5`

### Group keys（45）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_intrinsics/camera
camera_model
camera_observations
camera_observations/color_images
camera_observations/depth_images
force_observations
force_observations/force_left_end_effector_align
force_observations/force_left_end_effector_raw
force_observations/force_right_end_effector_align
force_observations/force_right_end_effector_raw
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_pose_align
puppet/end_effector_left_pose_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_pose_align
puppet/end_effector_right_pose_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
puppet/head_pose_align
puppet/head_pose_raw
puppet/head_position_align
puppet/head_position_raw
```

### Dataset keys（121）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_head` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_head` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_head` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 10 | `camera_extrinsics/camera_arm_right` | `[4, 4]` | `float64` |
| 11 | `camera_extrinsics/camera_left_arm_left` | `[4, 4]` | `float64` |
| 12 | `camera_extrinsics/camera_right_arm_right` | `[4, 4]` | `float64` |
| 13 | `camera_intrinsics/camera/dist_coeffs` | `[5]` | `float64` |
| 14 | `camera_intrinsics/camera/matrix` | `[3, 3]` | `float64` |
| 15 | `camera_model/camera_head` | `[]` | `vlen_string` |
| 16 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 17 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 18 | `camera_observations/color_images/camera_head` | `["T"]` | `vlen_uint8` |
| 19 | `camera_observations/color_images/camera_left` | `["T"]` | `vlen_uint8` |
| 20 | `camera_observations/color_images/camera_right` | `["T"]` | `vlen_uint8` |
| 21 | `camera_observations/depth_images/camera_head` | `["T"]` | `vlen_uint8` |
| 22 | `camera_observations/depth_images/camera_left` | `["T"]` | `vlen_uint8` |
| 23 | `camera_observations/depth_images/camera_right` | `["T"]` | `vlen_uint8` |
| 24 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 25 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 26 | `force_observations/force_left_end_effector_align/data` | `["T", 6]` | `float64` |
| 27 | `force_observations/force_left_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 28 | `force_observations/force_left_end_effector_align/timestamp` | `["T"]` | `float64` |
| 29 | `force_observations/force_left_end_effector_raw/data` | `["N", 6]` | `float64` |
| 30 | `force_observations/force_left_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 31 | `force_observations/force_left_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 32 | `force_observations/force_right_end_effector_align/data` | `["T", 6]` | `float64` |
| 33 | `force_observations/force_right_end_effector_align/is_intervene` | `["T"]` | `bool` |
| 34 | `force_observations/force_right_end_effector_align/timestamp` | `["T"]` | `float64` |
| 35 | `force_observations/force_right_end_effector_raw/data` | `["N", 6]` | `float64` |
| 36 | `force_observations/force_right_end_effector_raw/is_intervene` | `["N"]` | `bool` |
| 37 | `force_observations/force_right_end_effector_raw/timestamp` | `["N"]` | `float64` |
| 38 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 39 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 40 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 41 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 42 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 43 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 44 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 45 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 46 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 47 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 48 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 49 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 50 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 51 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 52 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 53 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 54 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 55 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 56 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 57 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 58 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 59 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 60 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 61 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 62 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 63 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 64 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 65 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 66 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 67 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 68 | `metadata/collection_time` | `[]` | `vlen_string` |
| 69 | `metadata/collector` | `[]` | `vlen_string` |
| 70 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 71 | `metadata/data_type` | `[]` | `vlen_string` |
| 72 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 73 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 74 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 75 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 76 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 77 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 78 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 79 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 80 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 81 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 82 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 83 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 84 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 85 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 86 | `puppet/end_effector_left_pose_align/data` | `["T", 7]` | `float32` |
| 87 | `puppet/end_effector_left_pose_align/is_intervene` | `["T"]` | `bool` |
| 88 | `puppet/end_effector_left_pose_align/timestamp` | `["T"]` | `float64` |
| 89 | `puppet/end_effector_left_pose_raw/data` | `["N", 7]` | `float32` |
| 90 | `puppet/end_effector_left_pose_raw/is_intervene` | `["N"]` | `bool` |
| 91 | `puppet/end_effector_left_pose_raw/timestamp` | `["N"]` | `float64` |
| 92 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 93 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 94 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 95 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 96 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 97 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 98 | `puppet/end_effector_right_pose_align/data` | `["T", 7]` | `float32` |
| 99 | `puppet/end_effector_right_pose_align/is_intervene` | `["T"]` | `bool` |
| 100 | `puppet/end_effector_right_pose_align/timestamp` | `["T"]` | `float64` |
| 101 | `puppet/end_effector_right_pose_raw/data` | `["N", 7]` | `float32` |
| 102 | `puppet/end_effector_right_pose_raw/is_intervene` | `["N"]` | `bool` |
| 103 | `puppet/end_effector_right_pose_raw/timestamp` | `["N"]` | `float64` |
| 104 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 105 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 106 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 107 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 108 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 109 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 110 | `puppet/head_pose_align/data` | `["T", 7]` | `float32` |
| 111 | `puppet/head_pose_align/is_intervene` | `["T"]` | `bool` |
| 112 | `puppet/head_pose_align/timestamp` | `["T"]` | `float64` |
| 113 | `puppet/head_pose_raw/data` | `["N", 7]` | `float32` |
| 114 | `puppet/head_pose_raw/is_intervene` | `["N"]` | `bool` |
| 115 | `puppet/head_pose_raw/timestamp` | `["N"]` | `float64` |
| 116 | `puppet/head_position_align/data` | `["T", 3]` | `float32` |
| 117 | `puppet/head_position_align/is_intervene` | `["T"]` | `bool` |
| 118 | `puppet/head_position_align/timestamp` | `["T"]` | `float64` |
| 119 | `puppet/head_position_raw/data` | `["N", 3]` | `float32` |
| 120 | `puppet/head_position_raw/is_intervene` | `["N"]` | `bool` |
| 121 | `puppet/head_position_raw/timestamp` | `["N"]` | `float64` |

## schema_4b13784f1e9b

- 文件数：1
- Group数：30
- Dataset数：74
- 工站：tienyi_9
- 任务：plug_cables
- Arm mode：dual
- Puppet pose：False
- Master pose：False
- Puppet joints：True
- Master joints：True
- Cameras：(无实际图像Dataset)
- 代表文件：`/media/jushen/project-rl-dataset/raw_data_0831download/tienyi_9/plug_cables/tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260818_left/success_episodes/0818_112229/data/trajectory.hdf5`

### Group keys（30）

```text
base_to_robot_transformation
camera_color_channel
camera_color_resolution
camera_depth_resolution
camera_extrinsics
camera_intrinsics
camera_model
camera_observations
camera_observations/color_images
master
master/arm_left_position_align
master/arm_left_position_raw
master/arm_right_position_align
master/arm_right_position_raw
master/end_effector_left_position_align
master/end_effector_left_position_raw
master/end_effector_right_position_align
master/end_effector_right_position_raw
master/waist_position_align
master/waist_position_raw
metadata
puppet
puppet/arm_left_position_align
puppet/arm_left_position_raw
puppet/arm_right_position_align
puppet/arm_right_position_raw
puppet/end_effector_left_position_align
puppet/end_effector_left_position_raw
puppet/end_effector_right_position_align
puppet/end_effector_right_position_raw
```

### Dataset keys（74）

| # | Key | Shape | Dtype |
|---:|---|---|---|
| 1 | `camera_color_channel/camera_left` | `[]` | `vlen_string` |
| 2 | `camera_color_channel/camera_right` | `[]` | `vlen_string` |
| 3 | `camera_color_channel/camera_top` | `[]` | `vlen_string` |
| 4 | `camera_color_resolution/camera_left` | `[2]` | `int64` |
| 5 | `camera_color_resolution/camera_right` | `[2]` | `int64` |
| 6 | `camera_color_resolution/camera_top` | `[2]` | `int64` |
| 7 | `camera_depth_resolution/camera_left` | `[2]` | `int64` |
| 8 | `camera_depth_resolution/camera_right` | `[2]` | `int64` |
| 9 | `camera_depth_resolution/camera_top` | `[2]` | `int64` |
| 10 | `camera_model/camera_left` | `[]` | `vlen_string` |
| 11 | `camera_model/camera_right` | `[]` | `vlen_string` |
| 12 | `camera_model/camera_top` | `[]` | `vlen_string` |
| 13 | `camera_observations/is_intervene` | `["T"]` | `bool` |
| 14 | `camera_observations/timestamp` | `["T"]` | `float64` |
| 15 | `master/arm_left_position_align/data` | `["T", 7]` | `float64` |
| 16 | `master/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 17 | `master/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 18 | `master/arm_left_position_raw/data` | `["N", 7]` | `float64` |
| 19 | `master/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 20 | `master/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 21 | `master/arm_right_position_align/data` | `["T", 7]` | `float64` |
| 22 | `master/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 23 | `master/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 24 | `master/arm_right_position_raw/data` | `["N", 7]` | `float64` |
| 25 | `master/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 26 | `master/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 27 | `master/end_effector_left_position_align/data` | `["T", 1]` | `float64` |
| 28 | `master/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 29 | `master/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 30 | `master/end_effector_left_position_raw/data` | `["N", 1]` | `float64` |
| 31 | `master/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 32 | `master/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 33 | `master/end_effector_right_position_align/data` | `["T", 1]` | `float64` |
| 34 | `master/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 35 | `master/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 36 | `master/end_effector_right_position_raw/data` | `["N", 1]` | `float64` |
| 37 | `master/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 38 | `master/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
| 39 | `master/waist_position_align/data` | `["T", 1]` | `float64` |
| 40 | `master/waist_position_align/is_intervene` | `["T"]` | `bool` |
| 41 | `master/waist_position_align/timestamp` | `["T"]` | `float64` |
| 42 | `master/waist_position_raw/data` | `["N", 1]` | `float64` |
| 43 | `master/waist_position_raw/is_intervene` | `["N"]` | `bool` |
| 44 | `master/waist_position_raw/timestamp` | `["N"]` | `float64` |
| 45 | `metadata/collection_time` | `[]` | `vlen_string` |
| 46 | `metadata/collector` | `[]` | `vlen_string` |
| 47 | `metadata/data_format_version` | `[]` | `vlen_string` |
| 48 | `metadata/data_type` | `[]` | `vlen_string` |
| 49 | `metadata/language_instruction` | `[]` | `vlen_string` |
| 50 | `metadata/xmigcs_version` | `[]` | `vlen_string` |
| 51 | `puppet/arm_left_position_align/data` | `["T", 7]` | `float32` |
| 52 | `puppet/arm_left_position_align/is_intervene` | `["T"]` | `bool` |
| 53 | `puppet/arm_left_position_align/timestamp` | `["T"]` | `float64` |
| 54 | `puppet/arm_left_position_raw/data` | `["N", 7]` | `float32` |
| 55 | `puppet/arm_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 56 | `puppet/arm_left_position_raw/timestamp` | `["N"]` | `float64` |
| 57 | `puppet/arm_right_position_align/data` | `["T", 7]` | `float32` |
| 58 | `puppet/arm_right_position_align/is_intervene` | `["T"]` | `bool` |
| 59 | `puppet/arm_right_position_align/timestamp` | `["T"]` | `float64` |
| 60 | `puppet/arm_right_position_raw/data` | `["N", 7]` | `float32` |
| 61 | `puppet/arm_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 62 | `puppet/arm_right_position_raw/timestamp` | `["N"]` | `float64` |
| 63 | `puppet/end_effector_left_position_align/data` | `["T", 1]` | `float32` |
| 64 | `puppet/end_effector_left_position_align/is_intervene` | `["T"]` | `bool` |
| 65 | `puppet/end_effector_left_position_align/timestamp` | `["T"]` | `float64` |
| 66 | `puppet/end_effector_left_position_raw/data` | `["N", 1]` | `float32` |
| 67 | `puppet/end_effector_left_position_raw/is_intervene` | `["N"]` | `bool` |
| 68 | `puppet/end_effector_left_position_raw/timestamp` | `["N"]` | `float64` |
| 69 | `puppet/end_effector_right_position_align/data` | `["T", 1]` | `float32` |
| 70 | `puppet/end_effector_right_position_align/is_intervene` | `["T"]` | `bool` |
| 71 | `puppet/end_effector_right_position_align/timestamp` | `["T"]` | `float64` |
| 72 | `puppet/end_effector_right_position_raw/data` | `["N", 1]` | `float32` |
| 73 | `puppet/end_effector_right_position_raw/is_intervene` | `["N"]` | `bool` |
| 74 | `puppet/end_effector_right_position_raw/timestamp` | `["N"]` | `float64` |
