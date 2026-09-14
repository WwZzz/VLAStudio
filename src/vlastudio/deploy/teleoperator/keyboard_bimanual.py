"""
Bimanual Keyboard Teleoperator for controlling two 6-DOF arms with grippers.

Outputs 14D delta actions: [left_7D, right_7D]
Each arm: [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]

Default Key Mappings:
- Left Arm (keyboard left half):
  - Translation: Q/E (X), A/D (Y), W/S (Z)
  - Roll: R/F, Pitch: 1/3, Yaw: Z/C
  - Gripper: V (open), B (close)

- Right Arm (keyboard right half):
  - Translation: U/O (X), J/L (Y), I/K (Z)
  - Roll: Y/H, Pitch: 7/9, Yaw: N/M
  - Gripper: . (open), / (close)

- Reset: 0
"""

from typing import Optional, List, Dict
from vlastudio.deploy.teleoperator.keyboard import Keyboard


class KeyboardBimanual(Keyboard):
    """
    Bimanual keyboard teleoperation for controlling two 6-DOF robot arms.
    
    Outputs 14D delta actions:
    - action[0:7]: Left arm [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
    - action[7:14]: Right arm [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
    
    Default Key Mappings:
    - Left Arm (left half of keyboard):
      - Translation: Q/E (X), D/A (Y), W/S (Z)
      - Roll: R/F, Pitch: 1/3, Yaw: Z/C
      - Gripper: V/B
    
    - Right Arm (right half of keyboard):
      - Translation: U/O (X), L/J (Y), I/K (Z)
      - Roll: Y/H, Pitch: 7/9, Yaw: N/M
      - Gripper: ./
    """

    def __init__(
        self,
        name: str = "keyboard_bimanual",
        max_size_mb: int = 1,
        fps: float = 50.0,
        trans_scale: float = 0.002,
        rot_scale: float = 0.01,
        gripper_scale: float = 0.01,
        left_key_mappings: Optional[List[Dict]] = None,
        right_key_mappings: Optional[List[Dict]] = None,
        reset_key: str = "0",
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize bimanual keyboard teleoperator
        
        Args:
            name: Name for shared memory
            max_size_mb: Max shared memory size in MB
            fps: Control frequency
            trans_scale: Scale factor for translation deltas (X, Y, Z)
            rot_scale: Scale factor for rotation deltas (Roll, Pitch, Yaw)
            gripper_scale: Scale factor for gripper delta
            left_key_mappings: Custom key mappings for left arm (7 mappings)
            right_key_mappings: Custom key mappings for right arm (7 mappings)
            reset_key: Key to reset all actions to zero
            debug: Enable debug output
        """
        # Build left arm key mappings (left half of keyboard)
        if left_key_mappings is None:
            left_key_mappings = [
                {"name": "L-X (fwd/back)", "key_positive": "q", "key_negative": "e", "scale": trans_scale},
                {"name": "L-Y (left/right)", "key_positive": "d", "key_negative": "a", "scale": trans_scale},
                {"name": "L-Z (up/down)", "key_positive": "w", "key_negative": "s", "scale": trans_scale},
                {"name": "L-Roll", "key_positive": "r", "key_negative": "f", "scale": rot_scale},
                {"name": "L-Pitch", "key_positive": "1", "key_negative": "3", "scale": rot_scale},
                {"name": "L-Yaw", "key_positive": "z", "key_negative": "c", "scale": rot_scale},
                {"name": "L-Gripper", "key_positive": "v", "key_negative": "b", "scale": gripper_scale},
            ]
        
        # Build right arm key mappings (right half of keyboard)
        if right_key_mappings is None:
            right_key_mappings = [
                {"name": "R-X (fwd/back)", "key_positive": "u", "key_negative": "o", "scale": trans_scale},
                {"name": "R-Y (left/right)", "key_positive": "l", "key_negative": "j", "scale": trans_scale},
                {"name": "R-Z (up/down)", "key_positive": "i", "key_negative": "k", "scale": trans_scale},
                {"name": "R-Roll", "key_positive": "y", "key_negative": "h", "scale": rot_scale},
                {"name": "R-Pitch", "key_positive": "7", "key_negative": "9", "scale": rot_scale},
                {"name": "R-Yaw", "key_positive": "n", "key_negative": "m", "scale": rot_scale},
                {"name": "R-Gripper", "key_positive": "period", "key_negative": "slash", "scale": gripper_scale},
            ]
        
        # Combine left and right mappings
        key_mappings = left_key_mappings + right_key_mappings
        
        super().__init__(
            name=name,
            max_size_mb=max_size_mb,
            fps=fps,
            title="Bimanual Keyboard Control (Left: QWEASD | Right: UIOJKL)",
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
        "type": "deploy.teleoperator.keyboard_bimanual.KeyboardBimanual",
        "name": "keyboard_bimanual",
        "args": {
            "name": "keyboard_bimanual",
            "fps": 50.0,
            "trans_scale": 0.002,
            "rot_scale": 0.01,
            "gripper_scale": 0.01,
            "debug": True,
        }
    }

    shm_name = device_config["args"]["name"]
    proc = mp.Process(target=start_device, args=(device_config,))
    proc.start()

    time.sleep(1.0)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Focus the GUI window and use keyboard to control.\n")
    print("Left Arm:  Q/E(X) D/A(Y) W/S(Z) R/F(Roll) 1/3(Pitch) Z/C(Yaw) V/B(Grip)")
    print("Right Arm: U/O(X) L/J(Y) I/K(Z) Y/H(Roll) 7/9(Pitch) N/M(Yaw) .//(Grip)")
    print()

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=10.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                left = arr[0:7]
                right = arr[7:14]
                print(
                    f"L: pos[{left[0]:+.3f},{left[1]:+.3f},{left[2]:+.3f}] "
                    f"rot[{left[3]:+.2f},{left[4]:+.2f},{left[5]:+.2f}] g:{left[6]:+.2f} | "
                    f"R: pos[{right[0]:+.3f},{right[1]:+.3f},{right[2]:+.3f}] "
                    f"rot[{right[3]:+.2f},{right[4]:+.2f},{right[5]:+.2f}] g:{right[6]:+.2f}",
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
