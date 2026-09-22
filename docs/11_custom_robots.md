# 11. Custom Robots

This guide explains how to add support for a new type of robot hardware, allowing you to run evaluations and collect data with your own physical systems.

## The `AbstractRobotInterface`

The core component for adding a new robot is the `AbstractRobotInterface`, defined in `deploy/robot/base.py`. You must create a new class that inherits from this base class and implements its abstract methods.

## Step 1: Create the Robot Interface Class

Create a new file in `deploy/robot/`, for example, `my_robot.py`.

```python
# In deploy/robot/my_robot.py
from .base import AbstractRobotInterface

class MyRobotInterface(AbstractRobotInterface):
    def __init__(self, config):
        self.config = config
        # Initialize the connection to the robot's hardware or drivers.
        # e.g., connect to ROS, initialize a vendor's SDK, etc.
        print("MyRobotInterface: Connecting...")

    def get_observation(self):
        """Get the latest sensor data from the robot."""
        # This method should block until a new observation is ready.
        raw_obs = {}
        # e.g., raw_obs['joint_positions'] = self.ros_subscriber.get_joint_states()
        # e.g., raw_obs['wrist_image'] = self.camera.get_frame()
        return raw_obs

    def publish_action(self, action):
        """Send a motor command to the robot."""
        # This method should be non-blocking.
        # e.g., self.ros_publisher.publish_joint_commands(action)
        pass

    def obs2meta(self, obs):
        """Convert raw sensor data into the standardized `MetaObs` format."""
        # The goal is to create a `MetaObs` object.
        from benchmark.base import MetaObs
        
        # Process raw_obs into the expected numpy arrays
        state_np = ...
        image_np = ... # Shape (K, C, H, W) where K is number of cameras
        
        return MetaObs(state=state_np, image=image_np)

    def meta2act(self, meta_action):
        """Convert a `MetaAction` from the policy into a raw motor command."""
        # Extract the numpy array from the MetaAction
        raw_action = meta_action.action
        # Potentially convert the action format if needed
        return raw_action

    def shutdown(self):
        """Safely disconnect from the robot."""
        print("MyRobotInterface: Shutting down.")
        # e.g., self.ros_connection.close()
```

## Step 2: Create the Configuration File

In `configs/robot/`, create a YAML file for your new robot, e.g., `my_robot.yaml`.

```yaml
# configs/robot/my_robot.yaml
class: "MyRobotInterface" # The name of your class in `deploy.robot.my_robot`
ros_topic_prefix: "/my_robot"
camera_serial_numbers:
  - "12345"
  - "67890"
# ... any other hardware-specific parameters your class needs
```

## Step 3: Integrate with the Factory

In `deploy/robot/base.py`, add your new class to the `make_robot` factory function.

```python
# In deploy/robot/base.py
def make_robot(config, args):
    # ... existing robot types
    elif config['class'] == 'MyRobotInterface':
        from .my_robot import MyRobotInterface
        return MyRobotInterface(config)
    # ...
```

## Step 4: Use the New Robot

You can now use your new robot in the real-world evaluation and data collection scripts.

```bash
# Data Collection
python start_teleop_recorder.py -c my_robot ...

# Real-World Evaluation
vlastudio infer --runtime current -r my_robot ...
```
