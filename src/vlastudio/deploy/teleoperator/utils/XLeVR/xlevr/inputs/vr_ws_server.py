"""
VR WebSocket server for receiving controller data from web browsers.
Adapted from the original vr_robot_teleop.py script.
"""

import asyncio
import json
import ssl
import websockets
import numpy as np
import math
import logging
from typing import Dict, Optional, Set
from scipy.spatial.transform import Rotation as R

from .base import BaseInputProvider, ControlGoal, ControlMode
from ..config import XLeVRConfig

logger = logging.getLogger(__name__)


class VRControllerState:
    """State tracking for a VR controller."""
    
    def __init__(self, hand: str):
        self.hand = hand
        self.grip_active = False
        self.trigger_active = False
        
        # Position tracking for relative movement
        self.origin_position = None
        self.origin_rotation = None
        
        # Quaternion-based rotation tracking (more stable than Euler)
        self.origin_quaternion = None
        self.accumulated_rotation_quat = None  # Accumulated rotation as quaternion
        
        # Rotation tracking for wrist control
        self.z_axis_rotation = 0.0  # For wrist_roll
        self.x_axis_rotation = 0.0  # For wrist_flex (pitch)
        
        # Position tracking
        self.current_position = None
        
        # Rotation tracking
        self.origin_wrist_angle = 0.0
    
    def reset_grip(self):
        """Reset grip state but preserve trigger state."""
        self.grip_active = False
        self.origin_position = None
        self.origin_rotation = None
        self.origin_quaternion = None
        self.accumulated_rotation_quat = None
        self.z_axis_rotation = 0.0
        self.x_axis_rotation = 0.0
    
    def reset_origin(self):
        """Reset origin position and rotation for auto-control mode."""
        self.origin_position = None
        self.origin_rotation = None
        self.origin_quaternion = None
        self.accumulated_rotation_quat = None
        self.z_axis_rotation = 0.0
        self.x_axis_rotation = 0.0


class VRWebSocketServer(BaseInputProvider):
    """WebSocket server for VR controller input."""
    
    def __init__(self, command_queue: asyncio.Queue, config: XLeVRConfig, print_only: bool = False):
        super().__init__(command_queue)
        self.config = config
        self.clients: Set = set()
        self.server = None
        self.print_only = print_only  # New flag for print-only mode
        self.latest_client_ip: Optional[str] = None
        
        # Controller states
        self.left_controller = VRControllerState("left")
        self.right_controller = VRControllerState("right")
        
        # Robot state tracking (for relative position calculation)
        self.left_arm_origin_position = None
        self.right_arm_origin_position = None
    
    def setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Setup SSL context for WebSocket server."""
        # Automatically generate SSL certificates if they don't exist
        if not self.config.ssl_files_exist:
            logger.info("SSL certificates not found for WebSocket server, attempting to generate them...")
            if not self.config.ensure_ssl_certificates():
                logger.error("Failed to generate SSL certificates for WebSocket server")
                return None
        
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(certfile=self.config.certfile, keyfile=self.config.keyfile)
            logger.info("SSL certificate and key loaded successfully for WebSocket server")
            return ssl_context
        except ssl.SSLError as e:
            logger.error(f"Error loading SSL cert/key: {e}")
            return None
    
    async def start(self):
        """Start the WebSocket server."""
        if not self.config.enable_vr:
            logger.info("VR WebSocket server disabled in configuration")
            return
        
        ssl_context = self.setup_ssl()
        if ssl_context is None:
            logger.error("Failed to setup SSL for WebSocket server")
            return
        
        host = self.config.host_ip
        port = self.config.websocket_port
        
        try:
            self.server = await websockets.serve(
                self.websocket_handler, 
                host, 
                port, 
                ssl=ssl_context
            )
            self.is_running = True
            logger.info(f"VR WebSocket server running on wss://{host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def stop(self):
        """Stop the WebSocket server."""
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("VR WebSocket server stopped")
    
    async def websocket_handler(self, websocket, path=None):
        """Handle WebSocket connections from VR controllers."""
        client_address = websocket.remote_address
        client_ip = client_address[0] if client_address else None
        self.latest_client_ip = client_ip
        logger.info(f"VR client connected: {client_address}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_controller_data(data, client_ip=client_ip)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
                except Exception as e:
                    logger.error(f"Error processing VR data: {e}")
                    # Add more context for debugging
                    logger.error(f"Data that caused error: {data}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
        
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"VR client {client_address} disconnected normally")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"VR client {client_address} disconnected with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with VR client {client_address}: {e}")
        finally:
            self.clients.discard(websocket)
            # Handle grip releases when client disconnects
            await self.handle_grip_release('left')
            await self.handle_grip_release('right')
            logger.info(f"VR client {client_address} cleanup complete")
    
    async def process_controller_data(self, data: Dict, client_ip: Optional[str] = None):
        """Process incoming VR controller data."""
        # Silent processing - debug output removed for cleaner logs
        
        # Process headset data if available (no print, just send goal for connection detection)
        if 'headset' in data:
            headset_data = data['headset']
            if headset_data and headset_data.get('position'):
                pos = headset_data['position']
                rot = headset_data.get('rotation', {})
                quat = headset_data.get('quaternion', {})
                
                # Create headset ControlGoal (silent, no print)
                headset_position = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
                headset_goal = ControlGoal(
                    arm="headset",
                    mode=ControlMode.POSITION_CONTROL,
                    target_position=headset_position,
                    wrist_roll_deg=rot.get('y', 0),  # Yaw rotation
                    wrist_flex_deg=rot.get('x', 0),   # Pitch rotation
                    metadata={
                        "source": "vr_headset",
                        "client_ip": client_ip,
                        "relative_position": False,
                        "vr_position": headset_position.tolist(),
                        "rotation": rot,
                        "quaternion": quat
                    }
                )
                await self.send_goal(headset_goal)
        
        # Process controller data
        if 'leftController' in data:
            await self.process_single_controller('left', data['leftController'])
        
        if 'rightController' in data:
            await self.process_single_controller('right', data['rightController'])
    
    async def process_single_controller(self, hand: str, data: Dict):
        """Process data for a single controller."""
        position = data.get('position', {})
        rotation = data.get('rotation', {})
        quaternion = data.get('quaternion', {})  # Get quaternion data directly
        grip_active = data.get('gripActive', False)
        trigger = data.get('trigger', 0)
        thumbstick = data.get('thumbstick', {})
        buttons = data.get('buttons', {})  # Get all button states
        
        # Also check squeeze button from buttons dict
        if buttons.get('squeeze', False):
            grip_active = True
        
        controller = self.left_controller if hand == 'left' else self.right_controller
        
        # Handle trigger for gripper control
        trigger_active = trigger > 0.5
        if trigger_active != controller.trigger_active:
            controller.trigger_active = trigger_active
            
            # Send gripper control goal - do not specify mode to avoid interfering with position control
            # Reverse behavior: gripper open by default, closes when trigger pressed
            gripper_goal = ControlGoal(
                arm=hand,
                gripper_closed=not trigger_active,  # Inverted: closed when trigger NOT active
                metadata={
                    "source": "vr_trigger",
                    "trigger": trigger,
                    "trigger_active": trigger_active,
                    "thumbstick": thumbstick
                }
            )
            await self.send_goal(gripper_goal)
            
            logger.info(f"🤏 {hand.upper()} gripper {'OPENED' if trigger_active else 'CLOSED'}")
        
        # 修改：直接响应控制器位置，不需要按squeeze键
        # 检查是否有位置数据
        if position and all(k in position for k in ['x', 'y', 'z']):
            # 如果还没有设置原点，设置当前位置为原点
            if controller.origin_position is None:
                controller.origin_position = np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
                
                # 设置四元数原点
                if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                    controller.origin_quaternion = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
                else:
                    controller.origin_quaternion = self.euler_to_quaternion(rotation) if rotation else None
                
                controller.accumulated_rotation_quat = controller.origin_quaternion
                controller.z_axis_rotation = 0.0
                controller.x_axis_rotation = 0.0
                
                # 发送重置信号
                reset_goal = ControlGoal(
                    arm=hand,
                    mode=ControlMode.POSITION_CONTROL,
                    target_position=None,
                    metadata={
                        "source": f"vr_auto_reset_{hand}",
                        "reset_target_to_current": True,
                        "trigger": trigger,
                        "trigger_active": trigger_active,
                        "thumbstick": thumbstick
                    }
                )
                await self.send_goal(reset_goal)
                logger.info(f"🎯 {hand.upper()} auto-activated - controlling {hand} arm")
            
            # 计算目标位置 - 改为绝对位置控制
            position_array = np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
            
            # 直接使用VR控制器的绝对位置，应用缩放
            absolute_position = position_array * self.config.vr_to_robot_scale
            
            # 计算手腕旋转
            if controller.origin_quaternion is not None:
                if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                    current_quat = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
                    self.update_quaternion_rotation_direct(controller, current_quat)
                else:
                    self.update_quaternion_rotation(controller, rotation)
                
                controller.z_axis_rotation = self.extract_roll_from_quaternion(controller.accumulated_rotation_quat, controller.origin_quaternion)
                controller.x_axis_rotation = self.extract_pitch_from_quaternion(controller.accumulated_rotation_quat, controller.origin_quaternion)
            
            # 创建绝对位置控制目标
            goal = ControlGoal(
                arm=hand,
                mode=ControlMode.POSITION_CONTROL,
                target_position=absolute_position,  # 绝对位置
                wrist_roll_deg=-controller.z_axis_rotation,
                wrist_flex_deg=-controller.x_axis_rotation,
                metadata={
                    "source": "vr_absolute_position",
                    "relative_position": False,  # 标记为绝对位置
                    "vr_position": position_array.tolist(),
                    "scaled_position": absolute_position.tolist(),
                    "trigger": trigger,
                    "trigger_active": trigger_active,
                    "thumbstick": thumbstick,
                    "grip_active": grip_active,  # Side button (squeeze) state
                    "buttons": buttons,  # All button states
                    "quaternion": {
                        "x": quaternion.get('x', 0) if quaternion else 0,
                        "y": quaternion.get('y', 0) if quaternion else 0,
                        "z": quaternion.get('z', 0) if quaternion else 0,
                        "w": quaternion.get('w', 1) if quaternion else 1,
                    } if quaternion else None
                }
            )
            await self.send_goal(goal)
        
        # 保留原有的squeeze键逻辑作为备用（可选）
        # 如果你想完全移除squeeze键控制，可以注释掉下面的代码
        """
        # Handle grip button for arm movement control (original logic)
        if grip_active:
            if not controller.grip_active:
                print_pose()
                # Grip just activated - set origin and reset target position
                controller.grip_active = True
                # Convert position dict to numpy array for proper subtraction later
                controller.origin_position = np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
                
                # Use quaternion data directly if available, otherwise fall back to Euler conversion
                if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                    controller.origin_quaternion = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
                    controller.origin_rotation = controller.origin_quaternion  # Store for compatibility
                else:
                    # Fallback to Euler angle conversion
                    controller.origin_quaternion = self.euler_to_quaternion(rotation) if rotation else None
                    controller.origin_rotation = controller.origin_quaternion
                
                controller.accumulated_rotation_quat = controller.origin_quaternion
                controller.z_axis_rotation = 0.0
                controller.x_axis_rotation = 0.0
                
                # Send reset signal to control loop to reset target position to current robot position
                reset_goal = ControlGoal(
                    arm=hand,
                    mode=ControlMode.POSITION_CONTROL,  # Keep in position control
                    target_position=None,  # Special signal
                    metadata={
                        "source": f"vr_grip_reset_{hand}",
                        "reset_target_to_current": True,  # Signal to reset target to current position
                        "trigger": trigger,
                        "trigger_active": trigger_active,
                        "thumbstick": thumbstick
                    }
                )
                await self.send_goal(reset_goal)
                
                logger.info(f"🔒 {hand.upper()} grip activated - controlling {hand} arm (target reset to current position)")
            
            # Compute target position
            if controller.origin_position is not None:
                # Convert position dict to numpy array for proper subtraction
                position_array = np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
                
                # Ensure origin_position is a numpy array
                if isinstance(controller.origin_position, dict):
                    # If origin_position is still a dict, convert it to numpy array
                    logger.warning(f"origin_position was dict, converting to numpy array for {hand} controller")
                    controller.origin_position = np.array([controller.origin_position.get('x', 0), controller.origin_position.get('y', 0), controller.origin_position.get('z', 0)])
                elif not isinstance(controller.origin_position, np.ndarray):
                    # If origin_position is neither dict nor numpy array, log warning and skip
                    logger.warning(f"origin_position is {type(controller.origin_position)}, skipping position calculation for {hand} controller")
                    return
                
                relative_delta = (position_array - controller.origin_position) * self.config.vr_to_robot_scale
                
                # Calculate Z-axis rotation for wrist_roll control
                # Calculate X-axis rotation for wrist_flex control
                if controller.origin_quaternion is not None:
                    # Update quaternion-based rotation tracking
                    if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                        # Use quaternion data directly
                        current_quat = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
                        self.update_quaternion_rotation_direct(controller, current_quat)
                    else:
                        # Fallback to Euler angle conversion
                        self.update_quaternion_rotation(controller, rotation)
                    
                    # Get accumulated rotations from quaternion
                    controller.z_axis_rotation = self.extract_roll_from_quaternion(controller.accumulated_rotation_quat, controller.origin_quaternion)
                    controller.x_axis_rotation = self.extract_pitch_from_quaternion(controller.accumulated_rotation_quat, controller.origin_quaternion)
                
                # Create position control goal
                # Note: We send relative position here, the control loop will handle
                # adding it to the robot's current position
                goal = ControlGoal(
                    arm=hand,
                    mode=ControlMode.POSITION_CONTROL,
                    target_position=relative_delta,  # Relative position delta
                    wrist_roll_deg=-controller.z_axis_rotation,
                    wrist_flex_deg=-controller.x_axis_rotation,
                    metadata={
                        "source": "vr_grip",
                        "relative_position": True,
                        "origin_position": controller.origin_position.tolist(),
                        "trigger": trigger,
                        "trigger_active": trigger_active,
                        "thumbstick": thumbstick
                    }
                )
                await self.send_goal(goal)
        """
    
    async def handle_grip_release(self, hand: str):
        """Handle grip release for a controller."""
        if hand == 'left':
            controller = self.left_controller
        elif hand == 'right':
            controller = self.right_controller
        else:
            return
        
        if controller.grip_active:
            controller.reset_grip()
            
            # Send idle goal to stop arm control
            goal = ControlGoal(
                arm=hand,
                mode=ControlMode.IDLE,
                metadata={
                    "source": "vr_grip_release",
                    "trigger": 0.0,
                    "trigger_active": False,
                    "thumbstick": {}
                }
            )
            await self.send_goal(goal)
            
            logger.info(f"🔓 {hand.upper()} grip released - arm control stopped")
    
    async def handle_trigger_release(self, hand: str):
        """Handle trigger release for a controller."""
        controller = self.left_controller if hand == 'left' else self.right_controller
        
        if controller.trigger_active:
            controller.trigger_active = False
            
            # Send gripper closed goal - reversed behavior: gripper closes when trigger released
            goal = ControlGoal(
                arm=hand,
                gripper_closed=True,  # Close gripper when trigger released
                metadata={
                    "source": "vr_trigger_release",
                    "trigger": 0.0,
                    "trigger_active": False,
                    "thumbstick": {}
                }
            )
            await self.send_goal(goal)
            
            logger.info(f"🤏 {hand.upper()} gripper CLOSED (trigger released)")
    
    def euler_to_quaternion(self, euler_deg: Dict[str, float]) -> np.ndarray:
        """Convert Euler angles in degrees to quaternion [x, y, z, w]."""
        euler_rad = [math.radians(euler_deg['x']), math.radians(euler_deg['y']), math.radians(euler_deg['z'])]
        rotation = R.from_euler('xyz', euler_rad)
        return rotation.as_quat()
    
    def update_quaternion_rotation(self, controller: VRControllerState, current_euler: dict):
        """Update quaternion-based rotation tracking."""
        if not current_euler:
            return
        
        # Convert current Euler to quaternion
        current_quat = self.euler_to_quaternion(current_euler)
        
        # Store current quaternion for accumulated rotation calculation
        controller.accumulated_rotation_quat = current_quat
    
    def update_quaternion_rotation_direct(self, controller: VRControllerState, current_quat: np.ndarray):
        """Update quaternion-based rotation tracking using quaternion data directly."""
        if current_quat is None:
            return
        
        # Store current quaternion for accumulated rotation calculation
        controller.accumulated_rotation_quat = current_quat
    
    def extract_roll_from_quaternion(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """Extract roll rotation around Z-axis from relative quaternion rotation."""
        if current_quat is None or origin_quat is None:
            return 0.0
        
        try:
            # Calculate relative rotation quaternion (from origin to current)
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()
            
            # Project the relative rotation onto the Z-axis (roll)
            # Get the rotation vector (axis-angle representation)
            rotvec = relative_rotation.as_rotvec()
            
            # The Z-component of the rotation vector represents rotation around Z-axis (roll)
            z_rotation_rad = rotvec[2]
            z_rotation_deg = -np.degrees(z_rotation_rad)
            
            return z_rotation_deg
        except Exception as e:
            logger.warning(f"Error extracting roll from quaternion: {e}")
            return 0.0
    
    def extract_pitch_from_quaternion(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """Extract pitch rotation around X-axis from relative quaternion rotation."""
        if current_quat is None or origin_quat is None:
            return 0.0
        
        try:
            # Calculate relative rotation quaternion (from origin to current)
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()
            
            # Project the relative rotation onto the X-axis (pitch)
            # Get the rotation vector (axis-angle representation)
            rotvec = relative_rotation.as_rotvec()
            
            # The X-component of the rotation vector represents rotation around X-axis (pitch)
            x_rotation_rad = rotvec[0]
            x_rotation_deg = np.degrees(x_rotation_rad)
            
            return x_rotation_deg
        except Exception as e:
            logger.warning(f"Error extracting pitch from quaternion: {e}")
            return 0.0
    
    async def send_goal(self, goal: ControlGoal):
        """Send a control goal to the command queue or print it if in print-only mode."""
        if self.print_only:
            # Print the ControlGoal in a formatted way
            print(f"\n🎮 ControlGoal:")
            print(f"   Arm: {goal.arm}")
            print(f"   Mode: {goal.mode}")
            if goal.target_position is not None:
                print(f"   Target Position: [{goal.target_position[0]:.3f}, {goal.target_position[1]:.3f}, {goal.target_position[2]:.3f}]")
            if goal.wrist_roll_deg is not None:
                print(f"   Wrist Roll: {goal.wrist_roll_deg:.1f}°")
            if goal.wrist_flex_deg is not None:
                print(f"   Wrist Flex: {goal.wrist_flex_deg:.1f}°")
            if goal.gripper_closed is not None:
                print(f"   Gripper: {'CLOSED' if goal.gripper_closed else 'OPEN'}")
            if goal.metadata:
                print(f"   Metadata: {goal.metadata}")
            print()
        else:
            # Use the parent class method to send to queue
            await super().send_goal(goal)
