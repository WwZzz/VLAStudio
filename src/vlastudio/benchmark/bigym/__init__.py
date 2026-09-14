"""
BiGym Environment Wrapper for ILStudio.

BiGym is a bimanual manipulation benchmark with an H1 humanoid robot.

Observation/Action Structure (with 4-DOF floating_base=True):
- observation.state (16 dims): [floating_base(4), left_arm(5), right_arm(5), grippers(2)]
- observation.state_arm (12 dims): [left_arm(5), right_arm(5), grippers(2)]
- action (16 dims): [floating_base(4), left_arm(5), right_arm(5), grippers(2)]

Note: The 4-DOF floating base includes [pelvis_x, pelvis_y, pelvis_z, pelvis_rz].
"""
import numpy as np
from ..base import MetaEnv, MetaObs, MetaAction

# =============================================================================
# Dimension Constants - H1 Robot with Floating Base
# =============================================================================

# Action dimensions with 4-DOF floating_base (16 total): [X, Y, Z, RZ]
ACTION_DIM_FLOATING_BASE_4DOF = 4  # [pelvis_x, pelvis_y, pelvis_z, pelvis_rz]
ACTION_DIM_FLOATING_BASE_3DOF = 3  # [pelvis_x, pelvis_y, pelvis_rz] (legacy)
ACTION_DIM_LEFT_ARM = 5            # [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist]
ACTION_DIM_RIGHT_ARM = 5           # [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist]
ACTION_DIM_GRIPPER = 2             # [left_gripper, right_gripper]
ACTION_DIM_ARMS = ACTION_DIM_LEFT_ARM + ACTION_DIM_RIGHT_ARM + ACTION_DIM_GRIPPER  # 12

# Default: 4-DOF floating base (matches downloaded demos)
ACTION_DIM_FLOATING_BASE = ACTION_DIM_FLOATING_BASE_4DOF  # 4
ACTION_DIM_TOTAL = ACTION_DIM_FLOATING_BASE + ACTION_DIM_ARMS  # 16

# Observation dimensions
OBS_DIM_QPOS = 29               # All joint positions
OBS_DIM_QVEL = 29               # All joint velocities
OBS_DIM_GRIPPER = 2             # [left_gripper, right_gripper]
OBS_DIM_FLOATING_BASE_4DOF = 4  # [x, y, z, rz] positions (4-DOF)
OBS_DIM_FLOATING_BASE_3DOF = 3  # [x, y, rz] positions (3-DOF legacy)
OBS_DIM_FLOATING_BASE = OBS_DIM_FLOATING_BASE_4DOF  # Default: 4-DOF
OBS_DIM_FLOATING_BASE_ACTIONS = 4  # accumulated [dx, dy, dz, drz]

# State dimensions for policy input
STATE_DIM_TOTAL = 16            # Matches action: [floating_base(4), arms(10), grippers(2)]
STATE_DIM_ARM = 12              # Arms only: [arms(10), grippers(2)]

# =============================================================================
# Available Tasks
# =============================================================================

ALL_TASKS = [
    'ReachTarget',
    'ReachTargetSingle', 
    'ReachTargetDual',
    'StackBlocks',
    'MovePlate',
    'MoveTwoPlates',
    'FlipCup',
    'PickBox',
    'PutCups',
    'TakeCups',
    'StackCups',
    'SaucepanToHob',
    'WallShelving',
    'DrawerTopOpen',
    'DrawerTopClose',
    'CabinetsDoorOpenLeft',
    'CabinetsDoorCloseLeft',
    'Dishwasher',
    'DishwasherClose',
    'DishwasherOpen',
    'DishwasherCloseTrays',
    'DishwasherOpenTrays',
    'DishwasherUnloadCups',
    'DishwasherUnloadCutlery',
    'DishwasherUnloadPlates',
    'DishwasherLoadCups',
    'DishwasherLoadCutlery',
    'DishwasherLoadPlates',
    'GroceriesStoreLower',
    'GroceriesStoreUpper',
]

# Task descriptions for language conditioning
TASK_DESC = {
    'ReachTarget': 'Reach the red target sphere with either hand.',
    'ReachTargetSingle': 'Reach the red target sphere with the left hand.',
    'ReachTargetDual': 'Reach both target spheres, one with each hand.',
    'StackBlocks': 'Stack the blocks on top of each other.',
    'MovePlate': 'Move the plate to the target location.',
    'MoveTwoPlates': 'Move both plates to their target locations.',
    'FlipCup': 'Flip the cup upside down.',
    'PickBox': 'Pick up the box.',
    'PutCups': 'Put the cups on the shelf.',
    'TakeCups': 'Take the cups from the shelf.',
    'StackCups': 'Stack the cups.',
    'SaucepanToHob': 'Move the saucepan to the hob.',
    'WallShelving': 'Place items on the wall shelf.',
    'DrawerTopOpen': 'Open the top drawer.',
    'DrawerTopClose': 'Close the top drawer.',
    'CabinetsDoorOpenLeft': 'Open the left cabinet door.',
    'CabinetsDoorCloseLeft': 'Close the left cabinet door.',
    'Dishwasher': 'Operate the dishwasher.',
    'DishwasherClose': 'Close the dishwasher door.',
    'DishwasherOpen': 'Open the dishwasher door.',
    'DishwasherCloseTrays': 'Close the dishwasher trays.',
    'DishwasherOpenTrays': 'Open the dishwasher trays.',
    'DishwasherUnloadCups': 'Unload cups from the dishwasher.',
    'DishwasherUnloadCutlery': 'Unload cutlery from the dishwasher.',
    'DishwasherUnloadPlates': 'Unload plates from the dishwasher.',
    'DishwasherLoadCups': 'Load cups into the dishwasher.',
    'DishwasherLoadCutlery': 'Load cutlery into the dishwasher.',
    'DishwasherLoadPlates': 'Load plates into the dishwasher.',
    'GroceriesStoreLower': 'Store groceries on the lower shelf.',
    'GroceriesStoreUpper': 'Store groceries on the upper shelf.',
}


def create_env(config):
    """Create a BiGym environment from config."""
    return BiGymEnv(config)


class BiGymEnv(MetaEnv):
    """BiGym environment wrapper for ILStudio.
    
    This wrapper:
    - Returns observation.state (15 dims) as the default state
    - Accepts 15-dim action input
    
    Config options:
        task: str - Task name (e.g., 'ReachTarget')
        control_frequency: int - Control frequency in Hz (default: 50)
        action_mode: str - 'joint_position' or 'torque' (default: 'joint_position')
        absolute: bool - Use absolute joint positions (default: True)
        floating_base: bool - Enable floating base control (default: True)
        cameras: list - Camera names to use (default: ['head', 'left_wrist', 'right_wrist'])
        camera_resolution: tuple - Camera resolution (default: (256, 256))
        render_mode: str - 'human' or 'rgb_array' (default: 'rgb_array')
        max_timesteps: int - Maximum episode length (default: 500)
        use_state_arm: bool - Use 12-dim arm-only state instead of 15-dim full state (default: False)
    """
    
    def __init__(self, config, *args):
        self.config = config
        self.task_name = config.task
        
        # Control settings
        self.ctrl_space = getattr(config, 'ctrl_space', 'joint')
        self.ctrl_type = getattr(config, 'ctrl_type', 'abs')
        self.max_timesteps = getattr(config, 'max_timesteps', 500)
        
        # BiGym specific settings
        self.control_frequency = getattr(config, 'control_frequency', 50)
        self.action_mode_type = getattr(config, 'action_mode', 'joint_position')
        self.absolute = getattr(config, 'absolute', True)
        self.floating_base = getattr(config, 'floating_base', True)
        
        # Camera settings
        self.camera_names = getattr(config, 'cameras', ['head', 'left_wrist', 'right_wrist'])
        self.camera_resolution = getattr(config, 'camera_resolution', (256, 256))
        self.render_mode = getattr(config, 'render_mode', 'rgb_array')
        
        # State mode: use full 16-dim state by default, or 12-dim arm-only state
        self.use_state_arm = getattr(config, 'use_state_arm', False)
        
        # Floating base DOF mode: 4-DOF by default (matches downloaded demos)
        self.floating_base_4dof = getattr(config, 'floating_base_4dof', True)
        
        # Language instruction
        self.raw_lang = TASK_DESC.get(self.task_name, f'Complete the {self.task_name} task.')
        
        # Create environment
        env = self._create_env()
        super().__init__(env)
        
        # Cache action/state dimensions based on floating base DOF
        if self.floating_base:
            fb_dim = ACTION_DIM_FLOATING_BASE_4DOF if self.floating_base_4dof else ACTION_DIM_FLOATING_BASE_3DOF
            self.action_dim = fb_dim + ACTION_DIM_ARMS
            self.state_dim = STATE_DIM_ARM if self.use_state_arm else (fb_dim + ACTION_DIM_ARMS)
        else:
            self.action_dim = ACTION_DIM_ARMS
            self.state_dim = STATE_DIM_ARM
    
    def _create_env(self):
        """Create the underlying BiGym environment."""
        from bigym.action_modes import JointPositionActionMode, TorqueActionMode, PelvisDof
        from bigym.utils.observation_config import ObservationConfig, CameraConfig
        
        task_cls = self._get_task_class(self.task_name)
        
        # Determine floating DOFs
        if self.floating_base_4dof:
            floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        else:
            floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]
        
        if self.action_mode_type == 'joint_position':
            action_mode = JointPositionActionMode(
                floating_base=self.floating_base,
                absolute=self.absolute,
                floating_dofs=floating_dofs if self.floating_base else None,
            )
        else:
            action_mode = TorqueActionMode(
                floating_base=self.floating_base
            )
        
        camera_configs = [
            CameraConfig(
                name=cam_name,
                resolution=self.camera_resolution,
                rgb=True,
                depth=False
            )
            for cam_name in self.camera_names
        ]
        
        observation_config = ObservationConfig(
            cameras=camera_configs,
            proprioception=True
        )
        
        env = task_cls(
            action_mode=action_mode,
            observation_config=observation_config,
            control_frequency=self.control_frequency,
            render_mode=self.render_mode if self.render_mode != 'rgb_array' else None,
        )
        
        return env
    
    def _get_task_class(self, task_name: str):
        """Get task class by name."""
        # Core reach target tasks
        from bigym.envs.reach_target import ReachTarget, ReachTargetSingle, ReachTargetDual
        
        task_map = {
            'ReachTarget': ReachTarget,
            'ReachTargetSingle': ReachTargetSingle,
            'ReachTargetDual': ReachTargetDual,
        }
        
        # Manipulation tasks (FlipCup, StackBlocks)
        try:
            from bigym.envs.manipulation import FlipCup, StackBlocks
            task_map.update({
                'FlipCup': FlipCup,
                'StackBlocks': StackBlocks,
            })
        except ImportError:
            pass
        
        # Move plates tasks
        try:
            from bigym.envs.move_plates import MovePlate, MoveTwoPlates
            task_map.update({
                'MovePlate': MovePlate,
                'MoveTwoPlates': MoveTwoPlates,
            })
        except ImportError:
            pass
        
        # Pick and place tasks
        try:
            from bigym.envs.pick_and_place import (
                PutCups, TakeCups, PickBox, SaucepanToHob
            )
            task_map.update({
                'PutCups': PutCups,
                'TakeCups': TakeCups,
                'PickBox': PickBox,
                'SaucepanToHob': SaucepanToHob,
            })
        except ImportError:
            pass
        
        # Cupboard tasks
        try:
            from bigym.envs.cupboards import (
                StackCups, WallShelving,
                DrawerTopOpen, DrawerTopClose, 
                CabinetsDoorOpenLeft, CabinetsDoorCloseLeft
            )
            task_map.update({
                'StackCups': StackCups,
                'WallShelving': WallShelving,
                'DrawerTopOpen': DrawerTopOpen,
                'DrawerTopClose': DrawerTopClose,
                'CabinetsDoorOpenLeft': CabinetsDoorOpenLeft,
                'CabinetsDoorCloseLeft': CabinetsDoorCloseLeft,
            })
        except ImportError:
            pass
        
        # Dishwasher tasks
        try:
            from bigym.envs.dishwasher import (
                Dishwasher, DishwasherClose, DishwasherOpen, 
                DishwasherCloseTrays, DishwasherOpenTrays
            )
            task_map.update({
                'Dishwasher': Dishwasher,
                'DishwasherClose': DishwasherClose,
                'DishwasherOpen': DishwasherOpen,
                'DishwasherCloseTrays': DishwasherCloseTrays,
                'DishwasherOpenTrays': DishwasherOpenTrays,
            })
        except ImportError:
            pass
        
        try:
            from bigym.envs.dishwasher_cups import DishwasherUnloadCups, DishwasherLoadCups
            task_map.update({
                'DishwasherUnloadCups': DishwasherUnloadCups,
                'DishwasherLoadCups': DishwasherLoadCups,
            })
        except ImportError:
            pass
        
        try:
            from bigym.envs.dishwasher_cutlery import DishwasherUnloadCutlery, DishwasherLoadCutlery
            task_map.update({
                'DishwasherUnloadCutlery': DishwasherUnloadCutlery,
                'DishwasherLoadCutlery': DishwasherLoadCutlery,
            })
        except ImportError:
            pass
        
        try:
            from bigym.envs.dishwasher_plates import DishwasherUnloadPlates, DishwasherLoadPlates
            task_map.update({
                'DishwasherUnloadPlates': DishwasherUnloadPlates,
                'DishwasherLoadPlates': DishwasherLoadPlates,
            })
        except ImportError:
            pass
        
        # Groceries tasks
        try:
            from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper
            task_map.update({
                'GroceriesStoreLower': GroceriesStoreLower,
                'GroceriesStoreUpper': GroceriesStoreUpper,
            })
        except ImportError:
            pass
        
        if task_name not in task_map:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(task_map.keys())}")
        
        return task_map[task_name]
    
    def meta2act(self, maction: MetaAction):
        """Convert MetaAction to BiGym action.
        
        Expects action with 15 dims: [floating_base(3), left_arm(5), right_arm(5), grippers(2)]
        Or 12 dims if use_state_arm=True: [left_arm(5), right_arm(5), grippers(2)]
        """
        action = maction['action']
        
        # If action is 12-dim and we need 15-dim, prepend zeros for floating_base
        if len(action) == ACTION_DIM_ARMS and self.floating_base:
            action = np.concatenate([np.zeros(ACTION_DIM_FLOATING_BASE, dtype=np.float32), action])
        
        return action
    
    def obs2meta(self, obs) -> MetaObs:
        """Convert BiGym observation to MetaObs.
        
        Returns state as 16-dim (or 12-dim if use_state_arm=True) matching action structure:
        - observation.state (16 dims with 4-DOF): [floating_base(4), left_arm(5), right_arm(5), grippers(2)]
        - observation.state_arm (12 dims): [left_arm(5), right_arm(5), grippers(2)]
        
        BiGym observation structure:
            - proprioception: (qpos + qvel) where qpos has 29 joints
            - proprioception_grippers: gripper states [left, right]
            - proprioception_floating_base: floating base position [x, y, z, rz] (4-DOF)
            - proprioception_floating_base_actions: accumulated floating base actions
            - rgb_{camera_name}: (C, H, W) RGB images
        """
        # Parse proprioception to extract arm qpos
        if 'proprioception' in obs:
            proprio = np.array(obs['proprioception']).astype(np.float32)
            n_joints = len(proprio) // 2
            qpos = proprio[:n_joints]
            # Arms are at indices 11:16 (left) and 16:21 (right) in qpos
            left_arm_qpos = qpos[11:16] if len(qpos) > 16 else np.zeros(5, dtype=np.float32)
            right_arm_qpos = qpos[16:21] if len(qpos) > 21 else np.zeros(5, dtype=np.float32)
        else:
            left_arm_qpos = np.zeros(5, dtype=np.float32)
            right_arm_qpos = np.zeros(5, dtype=np.float32)
        
        # Gripper states
        if 'proprioception_grippers' in obs:
            gripper_state = np.array(obs['proprioception_grippers']).astype(np.float32)
        else:
            gripper_state = np.zeros(OBS_DIM_GRIPPER, dtype=np.float32)
        
        # Floating base position (4-DOF: [x, y, z, rz] or 3-DOF: [x, y, rz])
        if 'proprioception_floating_base' in obs:
            floating_base = np.array(obs['proprioception_floating_base']).astype(np.float32)
        else:
            fb_dim = OBS_DIM_FLOATING_BASE_4DOF if self.floating_base_4dof else OBS_DIM_FLOATING_BASE_3DOF
            floating_base = np.zeros(fb_dim, dtype=np.float32)
        
        # Build state based on mode
        if self.use_state_arm:
            # 12-dim state: [left_arm(5), right_arm(5), grippers(2)]
            state = np.concatenate([
                left_arm_qpos,
                right_arm_qpos,
                gripper_state,
            ]).astype(np.float32)
        else:
            # 16-dim state (4-DOF) or 15-dim (3-DOF): [floating_base, left_arm(5), right_arm(5), grippers(2)]
            state = np.concatenate([
                floating_base,
                left_arm_qpos,
                right_arm_qpos,
                gripper_state,
            ]).astype(np.float32)
        
        # Extract images
        images = []
        for cam_name in self.camera_names:
            rgb_key = f'rgb_{cam_name}'
            if rgb_key in obs:
                images.append(obs[rgb_key])
        
        if images:
            image = np.stack(images, axis=0)
        else:
            image = None
        
        # Extract depth if available
        depth = None
        depth_images = []
        for cam_name in self.camera_names:
            depth_key = f'depth_{cam_name}'
            if depth_key in obs:
                depth_images.append(obs[depth_key])
        
        if depth_images:
            depth = np.stack(depth_images, axis=0)
        
        return MetaObs(
            state=state,
            image=image,
            depth=depth,
            raw_lang=self.raw_lang
        )
    
    def step(self, *args, **kwargs):
        """Execute one step in the environment.
        
        Args:
            maction: MetaAction with 'action' key containing 15-dim (or 12-dim) action
        
        Returns:
            meta_obs: MetaObs with 15-dim (or 12-dim) state
            reward: float
            success: bool
            info: dict
        """
        maction = args[0]
        action = self.meta2act(maction)
        
        # Clip action to valid bounds (BiGym has strict boundary checks)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        meta_obs = self.obs2meta(obs)
        
        success = info.get('task_success', 0.0) > 0.5
        
        info['terminated'] = terminated
        info['truncated'] = truncated
        info['success'] = success
        
        return meta_obs, reward, success, info
    
    def reset(self, seed=None):
        """Reset the environment.
        
        Returns:
            meta_obs: MetaObs with 15-dim (or 12-dim) state
        """
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        
        return self.obs2meta(obs)
    
    def close(self):
        """Close the environment."""
        try:
            self.env.close()
        except Exception:
            pass  # Ignore EGL cleanup errors
    
    @property
    def observation_space(self):
        """Return observation space info."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'cameras': self.camera_names,
            'camera_resolution': self.camera_resolution,
        }
    
    @property
    def action_space_dim(self):
        """Return action space dimension."""
        return self.action_dim
