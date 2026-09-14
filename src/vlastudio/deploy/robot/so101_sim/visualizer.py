"""
SO101 Simulation Visualizer

MuJoCo-based visualization for SO101 simulation robot.
Reads robot state from shared memory and renders the robot in a MuJoCo viewer.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from typing import Optional

from vlastudio.deploy.simulation.mujoco.base import prepare_mujoco_xml_path
from vlastudio.deploy.visualizer.base import BaseVisualizer


# Joint names in MuJoCo model (must match robot.py)
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def draw_coordinate_frame(viewer, pos, axis_length=0.05, axis_radius=0.002):
    """
    Draw RGB coordinate axes at the specified position.
    
    This draws the CONTROL coordinate system (mapped to MuJoCo world):
    - Red (Control X / W,S keys): Forward = MuJoCo +Y
    - Green (Control Y / A,D keys): Left/Right = MuJoCo -X  
    - Blue (Control Z / R,F keys): Up/Down = MuJoCo +Z
    
    Args:
        viewer: MuJoCo viewer object
        pos: 3D position of the frame origin
        axis_length: Length of each axis
        axis_radius: Radius of each axis cylinder
    """
    # Control frame to MuJoCo world frame mapping (empirically determined):
    # Control X (forward, W/S) -> MuJoCo +Y
    # Control Y (left, A/D)    -> MuJoCo -X
    # Control Z (up, R/F)      -> MuJoCo +Z
    axes_and_colors = [
        (np.array([0, 1, 0]), [1, 0, 0, 1]),   # Control X (forward) - Red -> MuJoCo +Y
        (np.array([-1, 0, 0]), [0, 1, 0, 1]),  # Control Y (left) - Green -> MuJoCo -X
        (np.array([0, 0, 1]), [0, 0, 1, 1]),   # Control Z (up) - Blue -> MuJoCo +Z
    ]
    
    with viewer.lock():
        for axis, color in axes_and_colors:
            # Calculate start and end points
            start = pos
            end = pos + axis * axis_length
            
            # Add geometry to scene
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),  # size (will be set by from-to)
                np.zeros(3),  # pos (will be set by from-to)
                np.eye(3).flatten(),  # mat
                np.array(color, dtype=np.float32)
            )
            # Set capsule from start to end
            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                axis_radius,
                start,
                end
            )
            viewer.user_scn.ngeom += 1


class Visualizer(BaseVisualizer):
    """
    MuJoCo Visualizer for SO101 Simulation Robot.
    
    Reads 'qpos' from shared memory and renders the robot state.
    """
    
    def __init__(self, 
                 shm_name: str, 
                 fps: float = 60.0,
                 xml_path: Optional[str] = None,
                 scene_name: Optional[str] = None,
                 scene_xml_path: Optional[str] = None,
                 show_tcp_frame: bool = True,
                 **kwargs):
        """
        Initialize the visualizer.
        
        Args:
            shm_name: Name of the shared memory to read robot data from
            fps: Target visualization frame rate
            xml_path: Path to MuJoCo XML model file (uses default if None)
            show_tcp_frame: Whether to show the TCP coordinate frame
        """
        super().__init__(shm_name=shm_name, fps=fps, **kwargs)
        
        self.xml_path = xml_path
        self.scene_name = scene_name
        self.scene_xml_path = scene_xml_path
        
        # MuJoCo objects
        self.mjmodel = None
        self.mjdata = None
        self.viewer = None
        self.qpos_indices = None
        self.tcp_site_id = None
        self.show_tcp_frame = show_tcp_frame
        self._generated_xml_path = None
    
    def setup(self) -> bool:
        """
        Load MuJoCo model and launch the viewer.
        """
        try:
            robot_xml_path = str(Path(__file__).parent / "mujoco_model" / "so_101.xml")
            self.xml_path, self._generated_xml_path = prepare_mujoco_xml_path(
                robot_xml_path,
                xml_path=self.xml_path,
                scene_name=self.scene_name,
                scene_xml_path=self.scene_xml_path,
                generated_name="so101_visualizer",
            )
            print(f"[Visualizer] Loading MuJoCo model from {self.xml_path}")
            self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
            self.mjdata = mujoco.MjData(self.mjmodel)
            
            # Get joint indices
            self.qpos_indices = np.array([
                self.mjmodel.jnt_qposadr[self.mjmodel.joint(name).id] 
                for name in JOINT_NAMES
            ])
            
            # Get TCP site index
            try:
                self.tcp_site_id = self.mjmodel.site('tcp').id
                print(f"[Visualizer] Found TCP site (id={self.tcp_site_id})")
            except KeyError:
                print("[Visualizer] Warning: TCP site not found in model")
                self.tcp_site_id = None
            
            # Launch passive viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.mjmodel, 
                self.mjdata,
                show_left_ui=True,
                show_right_ui=True,
            )
            
            print("[Visualizer] MuJoCo viewer launched")
            if self.show_tcp_frame:
                print("[Visualizer] TCP frame visualization enabled (RGB = XYZ)")
            return True
            
        except Exception as e:
            print(f"[Visualizer] Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize(self, data: dict) -> bool:
        """
        Update the visualization with new robot state.
        
        Args:
            data: Data from shared memory, expected to contain 'qpos'
            
        Returns:
            True to continue, False if viewer was closed
        """
        # Check if viewer is still running
        if self.viewer is None or not self.viewer.is_running():
            return False
        
        # Get qpos from data
        qpos = data.get('qpos', None)
        if qpos is None:
            return True  # No qpos data, but continue
        
        qpos = np.array(qpos)
        if len(qpos) != 6:
            return True  # Invalid qpos, but continue
        
        # Update MuJoCo state
        self.mjdata.qpos[self.qpos_indices] = qpos
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        
        # Reset user scene geometry count
        self.viewer.user_scn.ngeom = 0
        
        # Draw TCP coordinate frame (world-aligned control frame)
        if self.show_tcp_frame and self.tcp_site_id is not None:
            tcp_pos = self.mjdata.site_xpos[self.tcp_site_id].copy()
            draw_coordinate_frame(self.viewer, tcp_pos,
                                  axis_length=0.06, axis_radius=0.003)
        
        # Sync viewer
        self.viewer.sync()
        
        return True
    
    def cleanup(self):
        """
        Close the viewer.
        """
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        self.mjmodel = None
        self.mjdata = None
        if self._generated_xml_path is not None:
            try:
                Path(self._generated_xml_path).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            self._generated_xml_path = None


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SO101 Visualizer")
    parser.add_argument("--shm", "-s", type=str, default="so101_sim_test",
                        help="Shared memory name to connect to")
    args = parser.parse_args()
    
    visualizer = Visualizer(shm_name=args.shm)
    visualizer.start()
