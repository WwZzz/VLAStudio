# Robot Configurations

This directory contains configuration files for various robots. PyBullet-specific robot configurations are organized in the `pybullet/` subdirectory, while other robot types remain in the main directory.

## Directory Structure

- `pybullet/` - PyBullet simulation robot configurations
- `*.yaml` - Other robot configurations (real robots, different simulators, etc.)

## Available Robot Configurations

### Manipulator Arms

#### 1. Franka Panda (Original)
- **File**: `pybullet/franka_panda_sim_pybullet_joint.yaml`
- **Action Dim**: 9 (7 joints + 2 gripper joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: 7-DOF manipulator with 2-finger gripper

#### 2. Franka Panda (Delta EE)
- **File**: `pybullet/franka_panda_sim_pybullet.yaml`
- **Action Dim**: 7 (3 position + 3 orientation + 1 gripper)
- **Action Dtype**: `float64`
- **Control Type**: Delta end-effector control
- **Description**: 7-DOF manipulator with delta position/orientation control

#### 3. Kuka IIWA (Joint Control)
- **File**: `pybullet/kuka_iiwa_sim_pybullet_joint.yaml`
- **Action Dim**: 7 (7 joints only)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: 7-DOF industrial manipulator arm

#### 4. Kuka IIWA (Delta EE)
- **File**: `pybullet/kuka_iiwa_sim_pybullet.yaml`
- **Action Dim**: 6 (3 position + 3 orientation, no gripper)
- **Action Dtype**: `float64`
- **Control Type**: Delta end-effector control
- **Description**: 7-DOF industrial manipulator with delta control

### Mobile Robots

#### 5. Racecar
- **File**: `pybullet/racecar_sim_pybullet.yaml`
- **Action Dim**: 6 (6 controllable joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: Autonomous racecar with 6 controllable joints

#### 6. Husky
- **File**: `pybullet/husky_sim_pybullet.yaml`
- **Action Dim**: 4 (4 controllable joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: Mobile ground robot with 4 controllable joints

### Quadruped Robots

#### 7. A1 (Unitree)
- **File**: `pybullet/a1_sim_pybullet.yaml`
- **Action Dim**: 12 (12 leg joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: Quadruped robot with 3 joints per leg (4 legs × 3 = 12)

#### 8. Laikago
- **File**: `pybullet/laikago_sim_pybullet.yaml`
- **Action Dim**: 12 (12 leg joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: Quadruped robot with 3 joints per leg

#### 9. Mini Cheetah
- **File**: `pybullet/mini_cheetah_sim_pybullet.yaml`
- **Action Dim**: 12 (12 leg joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: MIT Mini Cheetah quadruped robot

#### 10. Aliengo
- **File**: `pybullet/aliengo_sim_pybullet.yaml`
- **Action Dim**: 12 (12 leg joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: Unitree Aliengo quadruped robot

### Gripper Only

#### 11. PR2 Gripper
- **File**: `pybullet/pr2_gripper_sim_pybullet.yaml`
- **Action Dim**: 2 (2 gripper joints)
- **Action Dtype**: `float64`
- **Control Type**: Absolute joint control
- **Description**: PR2 robot gripper only

## Usage Examples

**Note**: All robot configurations now include `action_dim` and `action_dtype` in the config file, so you don't need to specify them via command line arguments!

### For Manipulator Arms (Joint Control):
```bash
python start_teleop_recorder.py --config configs/robots/pybullet/kuka_iiwa_sim_pybullet_joint.yaml --shm_name tmp
```

### For Manipulator Arms (Delta EE Control):
```bash
python start_teleop_recorder.py --config configs/robots/pybullet/kuka_iiwa_sim_pybullet.yaml --shm_name tmp
```

### For Quadruped Robots:
```bash
python start_teleop_recorder.py --config configs/robots/pybullet/a1_sim_pybullet.yaml --shm_name tmp
```

### For Mobile Robots:
```bash
python start_teleop_recorder.py --config configs/robots/pybullet/racecar_sim_pybullet.yaml --shm_name tmp
```

### Command Line Override:
If you want to override the config file values, you can still specify them via command line:
```bash
python start_teleop_recorder.py --config configs/robots/pybullet/a1_sim_pybullet.yaml --action_dim 6 --action_dtype float32 --shm_name tmp
```

## Configuration Format

All robot configurations now use a **consistent format** that matches the controller configuration style:

```yaml
target: deploy.robot.sim_pybullet.AbsJointRobot
action_dim: 7
action_dtype: float64
robot_urdf_path: "kuka_iiwa/model.urdf"
initial_joint_positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
gripper_joint_indices: []  # No gripper
gripper_width: 0.0
use_gui: true
```

**Key Features:**
- All parameters are at the root level (no nested `config` section)
- `action_dim` and `action_dtype` are included in the config file
- Parameters are passed directly to the robot constructor
- Consistent with controller configuration format

## Notes

- **Action Dtype**: All robots use `float64` for high precision control
- **Initial Joint Positions**: Set to reasonable default values for each robot type
- **Gripper Control**: Only manipulator arms have gripper control; mobile and quadruped robots don't
- **Control Types**: 
  - `AbsJointRobot`: Direct joint angle control
  - `DeltaEERobot`: Delta end-effector position/orientation control (manipulators only)

## Adding New Robots

To add a new robot:
1. Check the robot's URDF path in PyBullet data
2. Count controllable joints using the script in the main directory
3. Create configuration file using the consistent format (all parameters at root level)
4. Test with the teleop recorder
