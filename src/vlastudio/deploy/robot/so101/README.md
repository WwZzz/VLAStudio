# So101 Robot Implementation

This directory contains the implementation for So101 robotic arms, following the same pattern as the Koch robot implementation.

## Overview

The So101 implementation provides:
- **Follower Robot**: `So101FollowerWithCamera` - Controls the So101 follower arm with integrated camera support
- **Leader Teleoperator**: `So101Leader` - Captures movements from a So101 leader arm for teleoperation

## Installation
```shell
cd third_party/lerobot
pip install -e ".[feetech]"
```

## Architecture

The implementation follows the standard robot interface pattern:

```
So101FollowerWithCamera (BaseRobot)
├── Camera integration (OpenCV)
├── Joint control (6 DOF)
├── Observation collection
└── Action execution

So101Leader (BaseTeleopDevice)  
├── Joint position reading
├── Action conversion
└── Communication management
```

## Configuration Files

### Robot Configurations
- `configs/robot/so101_follower.yaml` - Linux configuration
- `configs/robot/so101_follower_win.yaml` - Windows configuration

### Teleop Configurations  
- `configs/teleop/so101_leader.yaml` - Linux configuration
- `configs/teleop/so101_leader_win.yaml` - Windows configuration

## Implementation Notes

### Current Status
This implementation uses the official lerobot support for So101 robots, providing a complete integration.

The implementation leverages:

1. **Official lerobot Support**: Uses `lerobot.robots.so101_follower.SO101Follower` and `lerobot.teleoperators.so101_leader.SO101Leader`
2. **Camera Integration**: Built-in camera support through lerobot's configuration system
3. **Standardized Interface**: Follows the same patterns as other robots in the codebase

### Configuration

Make sure to adjust the following in the config files:
1. **Communication Ports**: Set the correct COM ports for your hardware
2. **Camera Settings**: Configure camera indices and resolution
3. **Robot IDs**: Use appropriate identifiers for your setup

## Usage

### Training Data Collection
```bash
# Start the follower robot
python start_teleop_recorder.py --robot_config configs/robot/so101_follower.yaml --task_config configs/task/your_task.yaml

# Start the leader for teleoperation
python start_teleop_controller.py --teleop_config configs/teleop/so101_leader.yaml
```

### Policy Evaluation
```bash
# Evaluate trained policy
python eval_real.py -r configs/robot/so101_follower.yaml --policy_config configs/policy/your_policy.yaml
```

## Hardware Requirements

- So101 Follower robot arm
- So101 Leader robot arm (for teleoperation)
- USB camera (for vision)
- Appropriate communication interfaces (USB, Serial, etc.)

## Troubleshooting

1. **Connection Issues**: Check COM port settings and hardware connections
2. **lerobot Installation**: Ensure lerobot is properly installed with So101 support
3. **Joint Mapping**: Verify motor names match your specific So101 configuration
4. **Permissions**: Ensure proper permissions for device access on Linux systems
