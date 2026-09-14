# SO101 Simulation Robot

A MuJoCo-based simulation robot for the SO-101 arm with end-effector control support.

## Features

- **End-Effector Control**: Accepts 7D delta actions (6-DOF pose + gripper)
- **Inverse Kinematics**: Converts end-effector targets to joint positions
- **MuJoCo Visualization**: Launches viewer in subprocess for real-time visualization
- **Shared Memory Integration**: Compatible with ILStudio's deploy framework

## Action Space

The robot accepts 7D delta actions:
- `action[0:3]`: Delta position (dx, dy, dz) in world frame
- `action[3:6]`: Delta orientation (d_roll, d_pitch, d_yaw)
- `action[6]`: Delta gripper position

Actions are scaled by configurable factors:
- `position_scale`: Scale for position deltas (default: 0.001)
- `rotation_scale`: Scale for rotation deltas (default: 0.01)
- `gripper_scale`: Scale for gripper deltas (default: 0.01)

## Observation Space

The robot provides observations containing:
- `qpos`: Joint positions [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
- `gpos`: End-effector pose [x, y, z, roll, pitch, yaw]

## Usage

```python
from deploy.robot.so101_sim import So101SimRobot

robot = So101SimRobot(
    name="so101_sim",
    use_viewer=True,
    position_scale=0.005,
    rotation_scale=0.02,
)

if robot.connect():
    # Send delta action
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move forward
    action_dict = robot.process_action({'action': action})
    robot.publish_action(action_dict['action'])
    
    # Get observation
    obs = robot.get_observation()
    print(f"qpos: {obs['qpos']}")
    print(f"gpos: {obs['gpos']}")
```

## Configuration

Example YAML configuration:

```yaml
type: deploy.robot.so101_sim.So101SimRobot
name: so101_sim
args:
  name: so101_sim
  max_size_mb: 64
  fps: 50.0
  control_shm_name: null
  use_viewer: true
  position_scale: 0.001
  rotation_scale: 0.01
  gripper_scale: 0.01
```

## Dependencies

- mujoco
- numpy
- scipy
- spatialmath-python
