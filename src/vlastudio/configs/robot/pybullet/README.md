# PyBullet Robot Configurations

This directory contains configuration files for robots available in PyBullet simulation. All configurations use the consistent format with parameters at the root level.

## Available Robots

### Manipulator Arms
- `franka_panda_sim_pybullet_joint.yaml` - Franka Panda (Joint Control, 9 DOF)
- `franka_panda_sim_pybullet.yaml` - Franka Panda (Delta EE Control, 7 DOF)
- `kuka_iiwa_sim_pybullet_joint.yaml` - Kuka IIWA (Joint Control, 7 DOF)
- `kuka_iiwa_sim_pybullet.yaml` - Kuka IIWA (Delta EE Control, 6 DOF)

### Mobile Robots
- `racecar_sim_pybullet.yaml` - Racecar (6 DOF)
- `husky_sim_pybullet.yaml` - Husky (4 DOF)

### Quadruped Robots
- `a1_sim_pybullet.yaml` - A1 (Unitree, 12 DOF)
- `laikago_sim_pybullet.yaml` - Laikago (12 DOF)
- `mini_cheetah_sim_pybullet.yaml` - Mini Cheetah (12 DOF)
- `aliengo_sim_pybullet.yaml` - Aliengo (12 DOF)

### Gripper Only
- `pr2_gripper_sim_pybullet.yaml` - PR2 Gripper (2 DOF)

## Usage

```bash
# Example: Kuka IIWA Joint Control
python start_teleop_recorder.py --config configs/robots/pybullet/kuka_iiwa_sim_pybullet_joint.yaml --shm_name tmp

# Example: A1 Quadruped
python start_teleop_recorder.py --config configs/robots/pybullet/a1_sim_pybullet.yaml --shm_name tmp
```

## Configuration Format

All configurations follow this consistent format:

```yaml
target: deploy.robot.sim_pybullet.AbsJointRobot
action_dim: 7
action_dtype: float64
robot_urdf_path: "kuka_iiwa/model.urdf"
initial_joint_positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
gripper_joint_indices: []
gripper_width: 0.0
use_gui: true
```

## Notes

- All robots use `float64` for high precision control
- Action dimensions are automatically read from config files
- PyBullet URDF paths are relative to PyBullet's data directory
- GUI is enabled by default for visualization
