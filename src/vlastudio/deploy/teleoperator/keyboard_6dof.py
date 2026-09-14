"""
Keyboard Teleoperator for 6-DOF End-Effector Control with Gripper
Inherits from Keyboard with preset key mappings for 6-DOF + gripper control.

Outputs 7D delta actions: [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]

Default Key Mappings:
- Translation: W/S (X), A/D (Y), R/F (Z)
- Roll: Q/E
- Pitch: T/G
- Yaw: Z/C
- Gripper: V (open), B (close)
- Reset: 0
"""

from typing import Optional, List, Dict
from vlastudio.deploy.teleoperator.keyboard import Keyboard


# Default key mappings for 6-DOF + gripper control
DEFAULT_6DOF_KEY_MAPPINGS = [
    {"name": "X (forward/back)", "key_positive": "w", "key_negative": "s", "scale": 1.0},
    {"name": "Y (left/right)", "key_positive": "a", "key_negative": "d", "scale": 1.0},
    {"name": "Z (up/down)", "key_positive": "r", "key_negative": "f", "scale": 1.0},
    {"name": "Roll", "key_positive": "q", "key_negative": "e", "scale": 1.0},
    {"name": "Pitch", "key_positive": "t", "key_negative": "g", "scale": 1.0},
    {"name": "Yaw", "key_positive": "z", "key_negative": "c", "scale": 1.0},
    {"name": "Gripper", "key_positive": "v", "key_negative": "b", "scale": 1.0},
]


class Keyboard6DOF(Keyboard):
    """
    Keyboard teleoperation for 6-DOF end-effector control with gripper.
    
    This is a specialized version of Keyboard with preset key mappings for
    controlling a 6-DOF robot arm with gripper.
    
    Outputs 7D delta actions:
    - action[0]: dx (forward/backward)
    - action[1]: dy (left/right)
    - action[2]: dz (up/down)
    - action[3]: d_roll
    - action[4]: d_pitch
    - action[5]: d_yaw
    - action[6]: d_gripper
    
    Default Key Mappings:
    - Translation: W/S (X), A/D (Y), R/F (Z)
    - Roll: Q/E
    - Pitch: T/G
    - Yaw: Z/C
    - Gripper: V (open), B (close)
    - Reset: 0
    
    You can override individual scales via config:
        trans_scale: Scale for translation (X, Y, Z)
        rot_scale: Scale for rotation (Roll, Pitch, Yaw)
        gripper_scale: Scale for gripper
    
    Or provide custom key_mappings to fully customize keys and scales.
    """

    def __init__(
        self,
        name: str = "keyboard_6dof",
        max_size_mb: int = 1,
        fps: float = 50.0,
        trans_scale: float = 1.0,
        rot_scale: float = 1.0,
        gripper_scale: float = 1.0,
        key_mappings: Optional[List[Dict]] = None,
        reset_key: str = "0",
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize Tkinter keyboard 6-DOF teleoperator
        
        Args:
            name: Name for shared memory
            max_size_mb: Max shared memory size in MB
            fps: Control frequency
            trans_scale: Scale factor for translation deltas (X, Y, Z)
            rot_scale: Scale factor for rotation deltas (Roll, Pitch, Yaw)
            gripper_scale: Scale factor for gripper delta
            key_mappings: Custom key mappings (overrides trans/rot/gripper_scale if provided)
            reset_key: Key to reset all actions to zero
            debug: Enable debug output
        """
        # Build key mappings with scales
        if key_mappings is None:
            key_mappings = [
                {"name": "X (forward/back)", "key_positive": "w", "key_negative": "s", "scale": trans_scale},
                {"name": "Y (left/right)", "key_positive": "a", "key_negative": "d", "scale": trans_scale},
                {"name": "Z (up/down)", "key_positive": "r", "key_negative": "f", "scale": trans_scale},
                {"name": "Roll", "key_positive": "q", "key_negative": "e", "scale": rot_scale},
                {"name": "Pitch", "key_positive": "t", "key_negative": "g", "scale": rot_scale},
                {"name": "Yaw", "key_positive": "z", "key_negative": "c", "scale": rot_scale},
                {"name": "Gripper", "key_positive": "v", "key_negative": "b", "scale": gripper_scale},
            ]
        
        super().__init__(
            name=name,
            max_size_mb=max_size_mb,
            fps=fps,
            title="6-DOF + Gripper Keyboard Control",
            key_mappings=key_mappings,
            reset_key=reset_key,
            debug=debug,
            **kwargs
        )


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import time
    import multiprocessing as mp
    from vlastudio.deploy.base import start_device
    from vlastudio.deploy.shm_utils import SharedMemoryChannel

    device_config = {
        "type": "deploy.teleoperator.keyboard_6dof.Keyboard6DOF",
        "name": "keyboard_6dof",
        "args": {
            "name": "keyboard_6dof",
            "fps": 50.0,
            "trans_scale": 1.0,
            "rot_scale": 1.0,
            "gripper_scale": 1.0,
            "debug": True,
        }
    }

    shm_name = device_config["args"]["name"]
    proc = mp.Process(target=start_device, args=(device_config,))
    proc.start()

    time.sleep(1.0)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Focus the GUI window and use keyboard to control.\n")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=10.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                print(
                    f"  pos: [{arr[0]:+.2f}, {arr[1]:+.2f}, {arr[2]:+.2f}] | "
                    f"rot: [{arr[3]:+.2f}, {arr[4]:+.2f}, {arr[5]:+.2f}] | "
                    f"grip: {arr[6]:+.2f}",
                    end="\r",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
        print("Done.")
