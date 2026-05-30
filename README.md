# rl_nav

Inference scripts for the RL navigation module of the [Unified Autonomy Stack](https://github.com/ntnu-arl/unified_autonomy_stack/). Full documentation at [ntnu-arl.github.io/unified_autonomy_stack/rl](https://ntnu-arl.github.io/unified_autonomy_stack/rl/).

ROS 2 package for reinforcement-learning-based LiDAR navigation on UAVs. Policies are trained in [Aerial Gym](https://github.com/ntnu-arl/aerial_gym_simulator) using [Sample Factory](https://github.com/alex-petrenko/sample-factory) (PPO + GRU) and deployed here for real-time inference.

The package focuses on **acceleration control**: the policy outputs body-frame linear accelerations and yaw rate directly from LiDAR and state observations, without an intermediate velocity controller.

## Demo Videos (Click to Play)

[![rl_nav demo 1](https://img.youtube.com/vi/l8Su8OXsM-E/0.jpg)](https://www.youtube.com/watch?v=l8Su8OXsM-E)

[![rl_nav demo 2](https://img.youtube.com/vi/bleQPb1kVI8/0.jpg)](https://www.youtube.com/watch?v=bleQPb1kVI8)

---

## Overview

| Item | Detail |
|------|--------|
| Control mode | Acceleration (body-frame ax, ay, az + yaw rate) |
| Observation | 17-dim state + 16×20 LiDAR grid (337 total) |
| Action | 4-dim continuous [-1, 1] → scaled to physical units |
| Network | CNN encoder → MLP (256-128-64) → GRU (128 hidden) |
| Training framework | [Sample Factory](https://github.com/alex-petrenko/sample-factory) (APPO) |
| Simulation | Aerial Gym (Isaac Gym) |

---

## Dependencies

```bash
pip install sample-factory torch scipy gymnasium ros2-numpy
```

`ros2-numpy` is only required by the real-robot node (`lidar_nav_acc_ros2_node.py`); the sim node does not need it.

ROS 2 packages: `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `mavros_msgs`, `std_msgs`

---

## Running the ROS 2 Node

The node is triggered by incoming LiDAR data and publishes acceleration commands at the same rate.

### Real robot

```bash
cd scripts/
python3 lidar_nav_acc_ros2_node.py \
    --env=lidar_navigation_task \
    --experiment=magpie_accel_policy4 \
    --train_dir=/path/to/rl_nav/models
```

### Simulation

```bash
cd scripts/
python3 lidar_nav_acc_ros2_sim_node.py \
    --env=lidar_navigation_task \
    --experiment=magpie_accel_policy4 \
    --train_dir=/path/to/rl_nav/models
```

The `--train_dir` flag must point to the `models/` directory of this repo (the parent of the experiment folder).

---

## ROS 2 Topics

### Subscribers

| Topic | Type | Description |
|-------|------|-------------|
| `/rmf_unipilot/lidar/points` | `sensor_msgs/PointCloud2` | Raw 3-D LiDAR scan |
| `/processed_lidar_up` | `std_msgs/Float32MultiArray` | Pre-processed 16×20 LiDAR grid (real robot fast path) |
| `/rmf_unipilot/odom` | `nav_msgs/Odometry` | Position, orientation, body-frame velocities |
| `/target` | `geometry_msgs/PoseStamped` | Navigation goal (position + yaw) |
| `/gbplanner_path` | `nav_msgs/Path` | Path from a planner — last pose is used as the target |
| `/reset` | `std_msgs/Empty` | Reset RNN hidden state |

### Publishers

| Topic | Type | Description |
|-------|------|-------------|
| `/rmf_unipilot/cmd/acc` | `geometry_msgs/Twist` | Acceleration command (linear = ax ay az, angular.z = yaw rate) |
| `/rmf_unipilot/cmd/acc_filtered` | `geometry_msgs/Twist` | EMA-filtered version of the above |
| `/mavros/setpoint_raw/local` | `mavros_msgs/PositionTarget` | MAVROS acceleration setpoint (FRAME_BODY_NED) |

---

## Observation Space

The policy receives a flat 337-dim vector at each step:

| Indices | Dims | Content |
|---------|------|---------|
| 0–2 | 3 | Unit vector to target in body frame (yaw-only rotation) |
| 3 | 1 | Distance to target (clamped to 7 m) |
| 4 | 1 | Roll |
| 5 | 1 | Pitch |
| 6 | 1 | Yaw error to target |
| 7–9 | 3 | Body-frame linear velocity |
| 10–12 | 3 | Body-frame angular velocity |
| 13–16 | 4 | Previous action |
| 17–336 | 320 | Flattened 16×20 downsampled LiDAR grid (inverted depth, 1/r) |

The LiDAR grid is produced by spherical binning (480 azimuth × 48 elevation) followed by min-pooling down to 16×20, then inverting depth.

---

## Action Space

The network outputs 4 values in [-1, 1] which are scaled to physical units:

| Channel | Scale (policy5) | Unit |
|---------|-----------------|------|
| ax (forward) | 2.0 × 0.85 | m/s² |
| ay (lateral) | 2.0 × 0.50 | m/s² |
| az (vertical) | 2.0 × 0.40 | m/s² |
| yaw rate | π/3 × 0.60 | rad/s |

---

## Changing the Model

To swap in a policy trained in Aerial Gym with [Sample Factory](https://github.com/alex-petrenko/sample-factory):

1. **Copy the experiment folder** from your Aerial Gym `train_dir` into `models/`:

   ```
   models/<your_experiment_name>/
   ├── config.json
   └── checkpoint_p0/
       └── best_*.pth
   ```

2. **Pass the experiment name** on the command line:

   ```bash
   python3 lidar_nav_acc_ros2_node.py \
       --env=lidar_navigation_task \
       --experiment=<your_experiment_name> \
       --train_dir=/path/to/rl_nav/models
   ```

3. **Match the action scaling** in `Config` at the top of the node script to whatever `action_transformation_function` your Aerial Gym task used. For example:

   ```python
   ACTION_SCALE = np.array([0.85, 0.5, 0.4, 0.6])  # policy4 values
   ```

4. **Match the EMA filter** if you changed dynamics:

   ```python
   ACTION_FILTER_ALPHA = np.array([0.25, 0.3, 0.001, 0.3])
   ```
   Alpha=0 means no filtering (pass-through). Alpha→1 means heavy smoothing.

The `config.json` inside the model folder is read automatically by Sample Factory to reconstruct the network architecture — do not edit it.

---

## Default Model

**`magpie_accel_policy4`** (Feb 2026) — stable acceleration policy tuned for the real Magpie platform. Use this as the starting point. Additional policies are archived under `models/` for reference.

---

## Config Class Reference

The `Config` class at the top of each node script controls all runtime parameters without touching the network:

```python
class Config:
    ACTION_SCALE   = np.array([0.85, 0.5, 0.4, 0.6])  # physical scaling per channel
    ACTION_FILTER_ALPHA = np.array([0.25, 0.3, 0.001, 0.3])  # EMA (0=off, 1=max smooth)
    LIDAR_MAX_RANGE = 10.0   # metres
    LIDAR_MIN_RANGE = 0.4    # metres
    MEDIAN_FILTER   = True
    MEDIAN_FILTER_KERNEL_SIZE = 7
    RESET_AT_NEW_WP = False   # reset RNN hidden state on each new waypoint
    DEVICE          = "cuda:0"
```

---

## Aerial Gym Training Reference

The task used for training is `lidar_navigation_task` in Aerial Gym. The Sample Factory entry point command was:

```bash
python train.py \
    --env=lidar_navigation_task \
    --experiment=<experiment_name> \
    --train_dir=<path>/models \
    --with_wandb=false
```

Network architecture fixed in `standalone_inference.py`:
- **Encoder**: Conv2d(1→16→32→64) on 16×20 LiDAR image, output 128-dim; concatenated with 17-dim state vector → MLP (256-128-64)
- **Core**: GRU with 128 hidden units, 1 layer
- **Head**: Linear to 4 actions with adaptive std


## Citation

If you use this code or the trained models in your research, please consider citing:

```
@misc{dharmadhikari2026unifiedautonomystackblueprint,
      title={The Unified Autonomy Stack: Toward a Blueprint for Generalizable Robot Autonomy}, 
      author={Mihir Dharmadhikari and Nikhil Khedekar and Mihir Kulkarni and Morten Nissov and Martin Jacquet and Angelos Zacharia and Marvin Harms and Albert Gassol Puigjaner and Philipp Weiss and Kostas Alexis},
      year={2026},
      eprint={2605.12735},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.12735}, 
}
```