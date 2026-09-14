import os
import yaml
from loguru import logger
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from .utils import resolve_yaml, parse_overrides, apply_overrides_to_mapping, apply_overrides_to_object, convert_yaml_string_types
from .training.loader import load_training_config
from data_utils.utils import _convert_to_type
from types import SimpleNamespace


class ConfigLoader:
    """Unified loader for training, task, policy, robot, teleop configs with CLI overrides."""

    def __init__(self, args=None, unknown_args=None):
        self.args = args
        self.unknown_args = unknown_args
        # Accept either a list of unknown args or a precomputed overrides dict
        if isinstance(self.unknown_args, dict):
            self._overrides = self.unknown_args
        else:
            self._overrides = parse_overrides(self.unknown_args or [])

    @staticmethod
    def _is_merged_dataset_config(cfg: Dict[str, Any]) -> bool:
        """Check if config is a merged dataset config.
        
        Merged dataset format: {merged_id: [{type:..., name:..., args:...}, ...]}
        
        A config is considered merged if:
        1. It has no 'type', 'name', or 'class' keys (not a regular dataset)
        2. It has exactly one key whose value is a list
        3. That list contains dataset configs (dicts with 'type' or 'name')
        """
        # Regular dataset configs have these keys
        if 'type' in cfg or 'name' in cfg or 'class' in cfg:
            return False
        
        # Check for merged format: single key with list value
        for key, value in cfg.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if the list contains dataset configs
                if isinstance(value[0], dict) and ('type' in value[0] or 'name' in value[0]):
                    return True
        
        return False
    
    @staticmethod
    def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize config to unified format (type, name, args).
        Supports both old and new formats for backward compatibility.
        
        Also supports merged dataset format: {merged_id: [{type:..., name:..., args:...}, ...]}
        
        Args:
            cfg: Original config dictionary
            
        Returns:
            Normalized config dictionary with type, name, and args
        """
        if not isinstance(cfg, dict):
            return cfg
        
        # Check if this is a merged dataset config: {merged_id: [sub_configs...]}
        # A merged config has exactly one key whose value is a list of dataset configs
        if ConfigLoader._is_merged_dataset_config(cfg):
            # Don't normalize merged dataset configs, return as-is
            return cfg
        
        normalized = {}
        
        # 1. Extract 'type' field from various sources
        if 'type' in cfg:
            normalized['type'] = cfg['type']
        elif 'module_path' in cfg:  # old policy format
            normalized['type'] = cfg['module_path']
        elif 'target' in cfg:  # old teleop/robot format
            normalized['type'] = cfg['target']
        elif 'manager_name' in cfg:  # old action_manager format
            normalized['type'] = cfg['manager_name']
        elif 'class' in cfg:  # old dataset format
            normalized['type'] = cfg['class']
        
        # 2. Extract 'name'
        if 'name' in cfg:
            normalized['name'] = cfg['name']
        
        # 3. Extract 'args' or collect parameters
        if 'args' in cfg:
            # Already in new format
            normalized['args'] = cfg['args']
        elif 'model_args' in cfg:
            # Old policy format
            normalized['args'] = cfg['model_args']
        else:
            # Collect non-reserved fields as args
            reserved_keys = {
                'type', 'name', 'args', 'module_path', 'target',
                'manager_name', 'class', 'model_args', 'pretrained_config',
                'config_class', 'model_class', 'data_processor',
                'data_collator', 'trainer_class', 'datasets', 'meta', 'envs',
                'visualizer',
            }
            args_dict = {k: v for k, v in cfg.items() if k not in reserved_keys}
            if args_dict:
                normalized['args'] = args_dict
        
        # 4. Preserve special fields
        special_fields = [
            'pretrained_config',
            'config_class',
            'model_class',
            'data_processor',
            'data_collator',
            'trainer_class',
            'datasets',
            'meta',
            'envs',
            'visualizer',
        ]
        for field in special_fields:
            if field in cfg:
                normalized[field] = cfg[field]
        
        # 5. Normalize datasets if present
        if 'datasets' in normalized:
            normalized['datasets'] = [
                ConfigLoader.normalize_config(ds) if isinstance(ds, dict) else ds 
                for ds in normalized['datasets']
            ]
        
        # 6. Normalize envs if present
        if 'envs' in normalized:
            normalized['envs'] = [
                ConfigLoader.normalize_config(env) if isinstance(env, dict) else env
                for env in normalized['envs']
            ]
        
        # 7. Preserve important top-level parameters that should not be in reserved_keys
        # These parameters may be defined at policy top-level and need to be accessible
        # for priority override logic (e.g., policy normalize > task normalize)
        important_params = ['action_normalize', 'state_normalize']
        for param in important_params:
            if param in cfg:
                normalized[param] = cfg[param]
        
        return normalized

    def get_overrides(self, category: str) -> Dict[str, Any]:
        return self._overrides.get(category, {})

    def _base_dir(self, category: str) -> str:
        base = Path(__file__).resolve().parent
        return str(base / category)

    def _resolve(self, category: str, name_or_path: str) -> str:
        try:
            return resolve_yaml(name_or_path, self._base_dir(category))
        except FileNotFoundError:
            return name_or_path

    def load_yaml_config(self, category: str, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        path = self._resolve(category, name_or_path)
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        
        # Convert string types in YAML (e.g., '1e-8' -> 1e-08)
        convert_yaml_string_types(cfg)
        
        # Apply command-line overrides
        overrides = self.get_overrides(category)
        if isinstance(cfg, list):
            # Multi-device robot/teleop YAML: --robot.args.foo applies to first dict entry.
            # Also supports --robot.0.args.foo / --robot.1.args.foo
            for dotted, raw in overrides.items():
                if raw is None:
                    continue
                keys = dotted.split(".")
                target = cfg
                if keys and keys[0].isdigit():
                    idx = int(keys[0])
                    if 0 <= idx < len(cfg) and isinstance(cfg[idx], dict):
                        apply_overrides_to_mapping(cfg[idx], {".".join(keys[1:]): raw}, _convert_to_type)
                    continue
                if cfg and isinstance(cfg[0], dict):
                    apply_overrides_to_mapping(cfg[0], {dotted: raw}, _convert_to_type)
        else:
            apply_overrides_to_mapping(cfg, overrides, _convert_to_type)
        
        return cfg, path

    def load_task(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        cfg, path = self.load_yaml_config('task', name_or_path)
        
        # Normalize dataset configs within the task
        if 'datasets' in cfg:
            cfg['datasets'] = [self.normalize_config(ds) for ds in cfg['datasets']]
        
        # Handle meta field - flatten to top level for backward compatibility
        if 'meta' in cfg and isinstance(cfg['meta'], dict):
            for key, value in cfg['meta'].items():
                if key not in cfg:
                    cfg[key] = value
        
        return cfg, path

    def load_policy(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        cfg, path = self.load_yaml_config('policy', name_or_path)
        
        # Normalize to unified format
        cfg = self.normalize_config(cfg)
        
        # Flatten args to top level for easier command line access
        # This allows --policy.camera_names, --policy.chunk_size, etc.
        if 'args' in cfg and isinstance(cfg['args'], dict):
            args_dict = cfg['args']
            # Create a flattened copy while preserving the original args
            flattened_cfg = cfg.copy()
            for key, value in args_dict.items():
                # Only add to top level if not already present at top level
                if key not in flattened_cfg:
                    flattened_cfg[key] = value
            cfg = flattened_cfg
            
            # Also preserve old 'model_args' for backward compatibility
            cfg['model_args'] = args_dict
            # Preserve old 'module_path' for backward compatibility
            if 'type' in cfg:
                cfg['module_path'] = cfg['type']
        else:
            # If no args dict, create model_args from top-level parameters
            if 'model_args' not in cfg:
                cfg['model_args'] = {}
        
        # IMPORTANT: Ensure top-level normalize parameters are available in model_args
        # This allows normalize parameters defined at policy top-level to override task settings
        for norm_key in ['action_normalize', 'state_normalize']:
            if norm_key in cfg:
                # Top-level normalize params should be in model_args for merge_all_parameters
                if norm_key not in cfg['model_args']:
                    cfg['model_args'][norm_key] = cfg[norm_key]
        
        return cfg, path

    def load_robot(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        cfg, path = self.load_yaml_config('robot', name_or_path)
        
        # Normalize to unified format
        cfg = self.normalize_config(cfg)
        
        # Flatten args to top level for backward compatibility
        if 'args' in cfg and isinstance(cfg['args'], dict):
            for key, value in cfg['args'].items():
                if key not in cfg:
                    cfg[key] = value
            # Preserve old 'target' field for backward compatibility
            if 'type' in cfg:
                cfg['target'] = cfg['type']
        
        return cfg, path

    def load_teleop(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        cfg, path = self.load_yaml_config('teleop', name_or_path)
        
        # Normalize to unified format
        cfg = self.normalize_config(cfg)
        
        # Flatten args to top level for backward compatibility
        if 'args' in cfg and isinstance(cfg['args'], dict):
            for key, value in cfg['args'].items():
                if key not in cfg:
                    cfg[key] = value
        
        # Preserve old 'target' field for backward compatibility
        if 'type' in cfg:
            cfg['target'] = cfg['type']
        
        return cfg, path

    def load_manager(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        """Load action manager config with support for command-line overrides via --manager.xxx"""
        cfg, path = self.load_yaml_config('action_manager', name_or_path)
        
        # Normalize to unified format
        cfg = self.normalize_config(cfg)
        
        # Flatten args to top level for backward compatibility
        if 'args' in cfg and isinstance(cfg['args'], dict):
            for key, value in cfg['args'].items():
                if key not in cfg:
                    cfg[key] = value
        
        # Preserve old 'manager_name' field for backward compatibility
        if 'type' in cfg:
            cfg['manager_name'] = cfg['type']
        
        return cfg, path

    def load_env(self, name_or_path: str) -> Tuple[Any, str]:
        """Load env config and return a namespace for attribute-style access.
        Supports both single environment and multiple environments (envs list).
        """
        cfg, path = self.load_yaml_config('env', name_or_path)
        
        # Check if it's a multi-env config
        if isinstance(cfg, list):
            # Already a list of envs
            normalized_envs = [self.normalize_config(env) for env in cfg]
        elif 'envs' in cfg and isinstance(cfg['envs'], list):
            # Has envs field
            normalized_envs = [self.normalize_config(env) for env in cfg['envs']]
        else:
            # Single env - normalize and wrap in list for consistency
            normalized_cfg = self.normalize_config(cfg)
            normalized_envs = [normalized_cfg]
        
        # Flatten args for each env to make them accessible at top level
        flattened_envs = []
        for env_cfg in normalized_envs:
            flat_env = env_cfg.copy()
            if 'args' in env_cfg and isinstance(env_cfg['args'], dict):
                # Flatten args to top level
                for key, value in env_cfg['args'].items():
                    if key not in flat_env:
                        flat_env[key] = value
            flattened_envs.append(flat_env)
        
        # Convert to namespace
        def to_ns(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [to_ns(x) for x in d]
            return d
        
        # Return single env or list based on original format
        if len(flattened_envs) == 1:
            return to_ns(flattened_envs[0]), path
        else:
            return [to_ns(env) for env in flattened_envs], path

    def load_training(self, name_or_path: str, hyper_args=None):
        """Return (training_config_obj, training_args_obj, resolved_path)."""
        path = self._resolve('training', name_or_path)
        
        # Use from_yaml to load with proper type conversions
        training_config = load_training_config(path)
        
        # Apply overrides to the config_dict (where parameters are actually stored)
        apply_overrides_to_mapping(training_config.config_dict, self.get_overrides('training'), _convert_to_type)
        
        if hyper_args is None:
            hyper_args = self.args
        training_args = training_config.to_training_arguments(hyper_args)
        return training_config, training_args, path

    # ===== Parameter merging (moved from parameter_merger.py) =====
    @staticmethod
    def calculate_image_sizes(camera_names: list, image_size: list) -> list:
        """Calculate image sizes for each camera using unified image_size.
        
        Args:
            camera_names: List of camera names
            image_size: Unified image size as [width, height]
            
        Returns:
            List of image sizes, one per camera (all the same now)
        """
        return [[int(i) for i in image_size] for _ in camera_names]

    @staticmethod
    def merge_all_parameters(task_config: Dict[str, Any], policy_config: Dict[str, Any], training_config: Any, args: Optional[Any] = None) -> Dict[str, Any]:
        """Merge task, policy, and training configurations with proper priority handling.
        
        Priority Rules:
        1. chunk_size: policy > task (policy overrides task datasets)
        2. action_normalize, state_normalize: policy > task
        3. action_norm_mask, state_norm_mask: task > policy (preserve task settings)
        4. action_dim, state_dim: task > policy (task overrides policy model_args)
        """
        task_params = {
            'action_dim': task_config.get('action_dim', 7),
            'state_dim': task_config.get('state_dim', 7),
            'camera_names': task_config.get('camera_names', ['primary']),
            'image_size': task_config.get('image_size', [256, 256]),
            'use_reasoning': task_config.get('use_reasoning', False),
            'use_prev_subtask': task_config.get('use_prev_subtask', False)
        }

        # Extract model_args and pretrained_config
        model_args = policy_config.get('model_args', {})
        pretrained_config = policy_config.get('pretrained_config', {})
        
        # ========== Priority Rule #1: chunk_size (policy > task) ==========
        # If policy has chunk_size and it differs from task datasets, override task datasets
        policy_chunk_size = model_args.get('chunk_size') or policy_config.get('chunk_size')
        if policy_chunk_size is not None and 'datasets' in task_config:
            for dataset in task_config['datasets']:
                # Check if this is a merged dataset: {merged_id: [sub_configs...]}
                if ConfigLoader._is_merged_dataset_config(dataset):
                    # Handle merged dataset: recursively override chunk_size in sub-datasets
                    for merged_id, sub_configs in dataset.items():
                        if isinstance(sub_configs, list):
                            for sub_config in sub_configs:
                                if isinstance(sub_config, dict) and 'args' in sub_config:
                                    if 'chunk_size' in sub_config['args']:
                                        task_chunk_size = sub_config['args']['chunk_size']
                                        if task_chunk_size != policy_chunk_size:
                                            sub_name = sub_config.get('name', 'unnamed')
                                            logger.warning(f"Config Override: chunk_size = {policy_chunk_size} (policy) overrides {task_chunk_size} (task.datasets['{merged_id}']['{sub_name}'])")
                                            sub_config['args']['chunk_size'] = policy_chunk_size
                elif 'args' in dataset and 'chunk_size' in dataset['args']:
                    # Regular dataset format
                    task_chunk_size = dataset['args']['chunk_size']
                    if task_chunk_size != policy_chunk_size:
                        logger.warning(f"Config Override: chunk_size = {policy_chunk_size} (policy) overrides {task_chunk_size} (task.datasets['{dataset.get('name', 'unnamed')}'])")
                        dataset['args']['chunk_size'] = policy_chunk_size
        
        # ========== Priority Rule #2: action_normalize, state_normalize (policy > task) ==========
        policy_action_norm = policy_config.get('action_normalize')
        policy_state_norm = policy_config.get('state_normalize')
        task_action_norm = task_config.get('action_normalize')
        task_state_norm = task_config.get('state_normalize')
        
        if policy_action_norm is not None and task_action_norm is not None and policy_action_norm != task_action_norm:
            logger.warning(f"Config Override: action_normalize = '{policy_action_norm}' (policy) overrides '{task_action_norm}' (task)")
        
        if policy_state_norm is not None and task_state_norm is not None and policy_state_norm != task_state_norm:
            logger.warning(f"Config Override: state_normalize = '{policy_state_norm}' (policy) overrides '{task_state_norm}' (task)")
        
        # ========== Priority Rule #3: action_norm_mask, state_norm_mask (task > policy) ==========
        # These should be preserved from task config (no override)
        task_action_norm_mask = task_config.get('action_norm_mask')
        task_state_norm_mask = task_config.get('state_norm_mask')
        
        # ========== Priority Rule #4: action_dim, state_dim (task > policy) ==========
        # Task dimensions override policy model_args
        task_action_dim = task_config.get('action_dim')
        task_state_dim = task_config.get('state_dim')
        policy_action_dim = model_args.get('action_dim')
        policy_state_dim = model_args.get('state_dim')
        
        if task_action_dim is not None and policy_action_dim is not None and task_action_dim != policy_action_dim:
            logger.warning(f"Config Override: action_dim = {task_action_dim} (task) overrides {policy_action_dim} (policy.model_args)")
            model_args['action_dim'] = task_action_dim
        
        if task_state_dim is not None and policy_state_dim is not None and task_state_dim != policy_state_dim:
            logger.warning(f"Config Override: state_dim = {task_state_dim} (task) overrides {policy_state_dim} (policy.model_args)")
            model_args['state_dim'] = task_state_dim
            
        # Also update top-level policy_config if these keys exist there (due to flattening)
        # This prevents the flattened old values from overriding the updated model_args later
        if 'action_dim' in policy_config and 'action_dim' in model_args:
            policy_config['action_dim'] = model_args['action_dim']
        if 'state_dim' in policy_config and 'state_dim' in model_args:
            policy_config['state_dim'] = model_args['state_dim']

        # Dynamically extract all model parameters from policy config
        # This includes both model_args and any flattened parameters from command line overrides
        model_params = {}
        
        # First add pretrained_config parameters
        if pretrained_config:
            model_params.update(pretrained_config)
        
        # Then add all model_args parameters
        if model_args:
            model_params.update(model_args)
        
        # Finally add any top-level parameters that were flattened (from command line overrides)
        # Skip known non-model parameters like 'name', 'module_path', 'model_args', 'pretrained_config'
        reserved_keys = {'name', 'module_path', 'model_args', 'pretrained_config', 'config_class', 'model_class', 'data_processor', 'data_collator', 'trainer_class'}
        for key, value in policy_config.items():
            if key not in reserved_keys and key not in model_params:
                model_params[key] = value
            elif key not in reserved_keys and key in model_params:
                # Top-level overrides win (command line overrides)
                model_params[key] = value

        # Dynamically extract all training parameters from training_config
        training_params = {}
        
        # Add preload_data (special parameter not part of TrainingArguments)
        training_params['preload_data'] = training_config.preload_data
        
        # Add all parameters from the config_dict (these will be passed to TrainingArguments)
        training_params.update(training_config.config_dict)

        cfg_params = policy_config.get('config_params', {}) if isinstance(policy_config, dict) else {}
        
        # ========== Apply Priority Rules to Preferred Values ==========
        
        # chunk_size: policy > task (already handled task datasets override above)
        preferred_chunk_size = (policy_config.get('chunk_size') or 
                              model_args.get('chunk_size') or 
                              cfg_params.get('chunk_size') or 
                              task_config.get('chunk_size', 16))
        
        # action_normalize, state_normalize: policy > task
        preferred_action_norm = (policy_config.get('action_normalize') or 
                               model_args.get('action_normalize') or 
                               cfg_params.get('action_normalize') or 
                               task_config.get('action_normalize', 'minmax'))
        preferred_state_norm = (policy_config.get('state_normalize') or 
                              model_args.get('state_normalize') or 
                              cfg_params.get('state_normalize') or 
                              task_config.get('state_normalize', 'minmax'))
        
        # action_norm_mask, state_norm_mask: task > policy (preserve task values)
        preferred_action_norm_mask = task_config.get('action_norm_mask')  # Task priority
        preferred_state_norm_mask = task_config.get('state_norm_mask')    # Task priority
        
        # camera_names: policy > task
        preferred_camera_names = policy_config.get('camera_names', None) 
        if preferred_camera_names is None:
            preferred_camera_names = model_args.get('camera_names', None) 
            if preferred_camera_names is None:
                preferred_camera_names = task_config.get('camera_names', [])

        all_params = {**task_params, **model_params, **training_params}
        all_params.update({
            'chunk_size': preferred_chunk_size,
            'action_normalize': preferred_action_norm,
            'state_normalize': preferred_state_norm,
            'camera_names': preferred_camera_names,  # Allow policy to override camera_names
        })
        
        # Add norm_mask parameters (task priority - only if they exist in task config)
        if preferred_action_norm_mask is not None:
            all_params['action_norm_mask'] = preferred_action_norm_mask
        if preferred_state_norm_mask is not None:
            all_params['state_norm_mask'] = preferred_state_norm_mask
        
        # ========== CRITICAL: Ensure task dimensions override everything ==========
        # After all merging, force task dimensions to take priority
        # This is the final enforcement of Priority Rule #4
        if task_action_dim is not None:
            all_params['action_dim'] = task_action_dim
        if task_state_dim is not None:
            all_params['state_dim'] = task_state_dim
        
        # Remove None values to avoid overriding existing values with None
        all_params = {k: v for k, v in all_params.items() if v is not None}

        if args is not None:
            for key, value in all_params.items():
                setattr(args, key, value)
            # Generate image_sizes for backward compatibility with policies that still use it
            if hasattr(args, 'image_size'):
                if isinstance(args.image_size, str):
                    args.image_size = eval(args.image_size)
                elif isinstance(args.image_size, int):
                    args.image_size = [args.image_size, args.image_size]
                args.image_sizes = ConfigLoader.calculate_image_sizes(args.camera_names, args.image_size)
            
            # IMPORTANT: Update args.model_args for policies that use it directly (like ACT)
            # This ensures the updated action_dim/state_dim are used during model initialization
            args.model_args = model_params.copy()
        return all_params


