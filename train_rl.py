"""
Reinforcement Learning Training Script

This script provides a unified interface for training RL algorithms.
It supports:
- Multiple registered algorithms (TD3, SAC, PPO, etc.)
- Vectorized environments for parallel data collection
- Flexible configuration via YAML files and CLI overrides
- Checkpoint saving and resuming

Usage:
    python train_rl.py -a td3 -e aloha --num_envs 4 --total_steps 1000000
    python train_rl.py -a configs/rl/td3.yaml -e configs/env/custom.yaml -o ckpt/td3_experiment
"""

import configs  # Must be first to suppress TensorFlow logs
import os
import argparse
import json
import importlib
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

from data_utils.utils import set_seed
from benchmark.utils import SequentialVectorEnv, organize_obs


def parse_args():
    """Parse command line arguments for RL training."""
    parser = argparse.ArgumentParser(description='Train an RL algorithm')
    
    # Algorithm arguments
    parser.add_argument('-a', '--algorithm', type=str, default='td3',
                       help='Algorithm name (registered) or path to YAML config file')
    
    # Environment arguments
    parser.add_argument('-e', '--env', type=str, default='aloha',
                       help='Env config (name under configs/env or absolute path to yaml)')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--use_subproc', action='store_true',
                       help='Use SubprocVectorEnv instead of SequentialVectorEnv')
    
    # Training arguments
    parser.add_argument('--total_steps', type=int, default=1000000,
                       help='Total environment steps for training')
    parser.add_argument('--start_steps', type=int, default=25000,
                       help='Number of random exploration steps before policy is used')
    parser.add_argument('--update_after', type=int, default=1000,
                       help='Number of steps before starting policy updates')
    parser.add_argument('--update_every', type=int, default=50,
                       help='Update policy every N environment steps')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for policy updates')
    parser.add_argument('--replay_size', type=int, default=1000000,
                       help='Replay buffer capacity')
    parser.add_argument('--expl_noise', type=float, default=0.1,
                       help='Exploration noise scale (can be overridden by algo config)')
    
    # Output and logging
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/rl_training',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluate policy every N steps')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--log_freq', type=int, default=1000,
                       help='Log training stats every N steps')
    
    # Misc
    parser.add_argument('-s', '--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args, unknown = parser.parse_known_args()
    args._unknown = unknown
    return args


def load_env_config(args):
    """Load environment configuration from YAML."""
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, '_unknown', []))
    env_cfg, env_cfg_path = cfg_loader.load_env(args.env)
    return env_cfg, env_cfg_path


def load_algo_config(args):
    """
    Load algorithm configuration from YAML.
    
    Priority:
    1. If path ends with .yaml/.yml, load directly
    2. If algorithm name (e.g., 'td3'), try to load from configs/rl/{name}.yaml
    3. If no config file found, use defaults
    """
    import yaml
    
    algo_name_or_path = args.algorithm
    
    # Check if it's a path to a YAML config file
    if algo_name_or_path.endswith('.yaml') or algo_name_or_path.endswith('.yml'):
        # Load from specified YAML config file
        if os.path.exists(algo_name_or_path):
            with open(algo_name_or_path, 'r') as f:
                algo_cfg = yaml.safe_load(f) or {}
            algo_cfg_path = algo_name_or_path
        else:
            # Try configs/rl directory
            config_path = os.path.join('configs/rl', algo_name_or_path)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    algo_cfg = yaml.safe_load(f) or {}
                algo_cfg_path = config_path
            else:
                raise FileNotFoundError(f"Algorithm config file not found: {algo_name_or_path}")
        
        logger.info(f"Loaded algorithm config from: {algo_cfg_path}")
        return algo_cfg, algo_cfg_path
    else:
        # Try to load from configs/rl/{algo_name}.yaml
        config_path = os.path.join('configs/rl', f'{algo_name_or_path}.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                algo_cfg = yaml.safe_load(f) or {}
            logger.info(f"Loaded algorithm config from: {config_path}")
            return algo_cfg, config_path
        else:
            # No config file, use defaults
            logger.info(f"No config file found for '{algo_name_or_path}', using default parameters")
            return {'type': algo_name_or_path}, None


def create_env_fn(env_cfg, env_module):
    """Create a factory function for environment creation."""
    def _create():
        return env_module.create_env(env_cfg)
    return _create


def create_vector_env(env_cfg, num_envs, use_subproc=False):
    """Create vectorized environment."""
    # Parse env type
    env_type = env_cfg.type
    if '.' in env_type:
        module_path, class_name = env_type.rsplit('.', 1)
        env_module = importlib.import_module(module_path)
        env_name = module_path.split('.')[-1] if '.' in module_path else module_path
    else:
        env_module = importlib.import_module(f"benchmark.{env_type}")
        env_name = env_type
    
    if not hasattr(env_module, 'create_env'):
        raise AttributeError(f"env module {env_type} has no 'create_env'")
    
    env_fns = [create_env_fn(env_cfg, env_module) for _ in range(num_envs)]
    
    if use_subproc and num_envs > 1:
        from tianshou.env import SubprocVectorEnv
        vec_env = SubprocVectorEnv(env_fns)
    else:
        vec_env = SequentialVectorEnv(env_fns)
    
    return vec_env, env_name, env_module


def get_env_dims(env_cfg, vec_env=None):
    """
    Get state and action dimensions from environment config.
    
    Priority:
    1. Read from env_cfg (state_dim, action_dim) - preferred
    2. Infer from vec_env if not specified in config - fallback
    
    Args:
        env_cfg: Environment configuration (namespace or dict)
        vec_env: Optional vectorized environment for fallback inference
    
    Returns:
        (state_dim, action_dim): Tuple of dimensions
    """
    state_dim = None
    action_dim = None
    
    # 1. Try to get from env_cfg first (preferred)
    if hasattr(env_cfg, 'state_dim'):
        state_dim = env_cfg.state_dim
        logger.info(f"Read state_dim={state_dim} from env config")
    if hasattr(env_cfg, 'action_dim'):
        action_dim = env_cfg.action_dim
        logger.info(f"Read action_dim={action_dim} from env config")
    
    # 2. Fallback: infer from environment if not specified
    if (state_dim is None or action_dim is None) and vec_env is not None:
        logger.warning("state_dim or action_dim not specified in env config, inferring from environment...")
        
        if state_dim is None:
            # Reset to get observation
            obs = vec_env.reset()
            
            # Handle different observation formats
            if hasattr(obs, 'state'):
                state_dim = obs.state.shape[-1] if hasattr(obs.state, 'shape') else len(obs.state)
            elif isinstance(obs, np.ndarray):
                if obs.ndim == 1:
                    state_dim = obs.shape[0]
                else:
                    # For vectorized env, obs[0] is single env obs
                    first_obs = obs[0] if obs.dtype == np.object_ else obs[0]
                    if hasattr(first_obs, 'state'):
                        state_dim = first_obs.state.shape[-1]
                    else:
                        state_dim = first_obs.shape[-1] if hasattr(first_obs, 'shape') else len(first_obs)
            elif isinstance(obs, dict):
                state_dim = obs.get('state', obs.get('observation')).shape[-1]
            else:
                # Try to get first env's observation
                first_obs = obs[0] if hasattr(obs, '__getitem__') else obs
                if hasattr(first_obs, 'state'):
                    state_dim = first_obs.state.shape[-1]
                else:
                    state_dim = len(first_obs) if hasattr(first_obs, '__len__') else 1
            logger.info(f"Inferred state_dim={state_dim} from environment")
        
        if action_dim is None:
            # Get action dimension from action_space if available
            single_env = vec_env.envs[0] if hasattr(vec_env, 'envs') else vec_env
            if hasattr(single_env, 'action_space'):
                action_dim = single_env.action_space.shape[0]
            elif hasattr(single_env, 'env') and hasattr(single_env.env, 'action_space'):
                action_dim = single_env.env.action_space.shape[0]
            else:
                # Default fallback
                logger.warning("Could not determine action_dim from env, defaulting to state_dim")
                action_dim = state_dim
            logger.info(f"Inferred action_dim={action_dim} from environment")
    
    if state_dim is None or action_dim is None:
        raise ValueError("Could not determine state_dim and action_dim. Please specify them in env config.")
    
    return state_dim, action_dim


def create_meta_policy(args, algo_cfg, state_dim, action_dim):
    """
    Create MetaPolicy for the algorithm.
    
    Similar to how load_policy() works in policy/utils.py.
    """
    from benchmark.base import MetaPolicy
    from data_utils.normalize import Identity
    
    # Get control space and type from config or defaults
    ctrl_space = 'ee'
    ctrl_type = 'delta'
    
    if algo_cfg is not None:
        ctrl_space = algo_cfg.get('ctrl_space', ctrl_space)
        ctrl_type = algo_cfg.get('ctrl_type', ctrl_type)
    
    # For RL, we typically use identity normalizers (actions are already in [-1, 1])
    action_normalizer = Identity()
    state_normalizer = Identity()
    
    # Store in args for later use
    args.ctrl_space = ctrl_space
    args.ctrl_type = ctrl_type
    
    return {
        'action_normalizer': action_normalizer,
        'state_normalizer': state_normalizer,
        'ctrl_space': ctrl_space,
        'ctrl_type': ctrl_type,
    }


def create_replay_buffer(args, state_dim, action_dim, num_envs):
    """Create replay buffer."""
    from rl.buffer import MetaReplay
    
    replay = MetaReplay(
        capacity=args.replay_size,
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=num_envs,
        device='cpu',  # Store on CPU, move to GPU during training
    )
    return replay


def create_algorithm(args, algo_cfg, state_dim, action_dim, replay, meta_policy_params):
    """
    Create RL algorithm from config dynamically.
    
    Args:
        args: Command line arguments
        algo_cfg: Algorithm config dict (from YAML)
        state_dim: State dimension
        action_dim: Action dimension
        replay: Replay buffer
        meta_policy_params: Dict with normalizers and ctrl settings for MetaPolicy
    """
    import dataclasses
    from rl.algorithms import get_algorithm_class, get_config_class, list_algorithms
    
    logger.info(f"Available algorithms: {list_algorithms()}")
    
    # Determine algorithm type from config
    algo_args = algo_cfg if algo_cfg is not None else {}
    algo_type = algo_args.get('type', algo_args.get('algorithm', args.algorithm))
    
    # Get algorithm class and config class dynamically
    AlgorithmClass = get_algorithm_class(algo_type)
    ConfigClass = get_config_class(algo_type)
    logger.info(f"Using algorithm: {AlgorithmClass.__name__}")
    
    # Build config if ConfigClass is available
    config = None
    if ConfigClass is not None:
        # Required parameters
        config_params = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'device': args.device,
        }
        
        # Get config field names and add matching parameters from algo_cfg
        if dataclasses.is_dataclass(ConfigClass):
            config_fields = {f.name for f in dataclasses.fields(ConfigClass) if f.name not in config_params}
            for key in config_fields:
                if key in algo_args:
                    config_params[key] = algo_args[key]
        
        config = ConfigClass(**config_params)
        
        # Log config (dynamically get attributes)
        log_attrs = ['discount', 'tau', 'actor_lr', 'critic_lr', 'lr']
        log_parts = [f"{attr}={getattr(config, attr)}" for attr in log_attrs if hasattr(config, attr)]
        if log_parts:
            logger.info(f"{AlgorithmClass.__name__} config: {', '.join(log_parts)}")
    
    # Get exploration noise from algo_cfg (overrides command line)
    if 'expl_noise' in algo_args:
        args.expl_noise = algo_args['expl_noise']
        logger.info(f"Using expl_noise={args.expl_noise} from algo config")
    
    # Create algorithm
    if config is not None:
        algorithm = AlgorithmClass(
            replay=replay,
            config=config,
            meta_policy=None,
            ctrl_space=meta_policy_params['ctrl_space'],
            ctrl_type=meta_policy_params['ctrl_type'],
        )
    else:
        # Fallback for algorithms without config class
        excluded_keys = {'type', 'algorithm', 'ctrl_space', 'ctrl_type', 'expl_noise'}
        filtered_args = {k: v for k, v in algo_args.items() if k not in excluded_keys}
        
        algorithm = AlgorithmClass(
            replay=replay,
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device,
            **filtered_args
        )
    
    return algorithm


def extract_state(obs, state_key='state'):
    """Extract state array from observation."""
    if hasattr(obs, state_key):
        return getattr(obs, state_key)
    elif isinstance(obs, dict):
        return obs.get(state_key, obs.get('observation'))
    elif isinstance(obs, np.ndarray):
        if obs.dtype == np.object_:
            # Array of MetaObs
            return np.stack([getattr(o, state_key) if hasattr(o, state_key) else o for o in obs])
        return obs
    return obs


def train(args):
    """Main training loop using OffPolicyTrainer and DummyCollector."""
    # Set seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Load environment config
    env_cfg, env_cfg_path = load_env_config(args)
    logger.info(f"Loaded env config from: {env_cfg_path}")
    
    # Load algorithm config (similar to how eval_sim.py loads policy)
    algo_cfg, algo_cfg_path = load_algo_config(args)
    
    # Create vectorized environment
    vec_env, env_name, env_module = create_vector_env(
        env_cfg, args.num_envs, args.use_subproc
    )
    logger.info(f"Created {args.num_envs} parallel environments: {env_name}")
    
    # Sync derived values from env config (like eval_sim.py does)
    if hasattr(env_cfg, 'max_timesteps'):
        args.max_timesteps = env_cfg.max_timesteps
    else:
        args.max_timesteps = 1000  # Default
    
    if hasattr(env_cfg, 'task'):
        args.task = env_cfg.task
    
    # Get environment dimensions from config (preferred) or infer from env (fallback)
    state_dim, action_dim = get_env_dims(env_cfg, vec_env)
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create MetaPolicy parameters (normalizers, ctrl settings)
    meta_policy_params = create_meta_policy(args, algo_cfg, state_dim, action_dim)
    logger.info(f"Control space: {args.ctrl_space}, Control type: {args.ctrl_type}")
    
    # Create replay buffer
    replay = create_replay_buffer(args, state_dim, action_dim, args.num_envs)
    logger.info(f"Created replay buffer with capacity {args.replay_size}")
    
    # Create algorithm (with MetaPolicy)
    algorithm = create_algorithm(args, algo_cfg, state_dim, action_dim, replay, meta_policy_params)
    algorithm.set_env(vec_env)  # Bind environment for action processing
    logger.info(f"Created algorithm: {algorithm}")
    
    # Create Collector for environment interaction
    from rl.collectors import DummyCollector
    collector = DummyCollector(
        envs=vec_env,
        algorithm=algorithm,
        ctrl_space=args.ctrl_space,
        action_dim=action_dim,
    )
    
    # Create Trainer configuration
    from rl.trainers.offpolicy_trainer import OffPolicyTrainer, OffPolicyTrainerConfig
    trainer_config = OffPolicyTrainerConfig(
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        batch_size=args.batch_size,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        max_timesteps=args.max_timesteps,
        ctrl_space=args.ctrl_space,
        expl_noise=args.expl_noise,
    )
    
    # Create evaluation environment factory
    def eval_env_fn():
        return create_env_fn(env_cfg, env_module)()
    
    # Select vector environment class based on args
    if args.use_subproc:
        from benchmark.utils import SubprocVectorEnv
        eval_vec_env_cls = SubprocVectorEnv
    else:
        eval_vec_env_cls = SequentialVectorEnv
    
    # Create Trainer
    trainer = OffPolicyTrainer(
        algorithm=algorithm,
        collector=collector,
        config=trainer_config,
        eval_env_fn=eval_env_fn,
        eval_vec_env_cls=eval_vec_env_cls,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.train(resume_step=start_step)
    
    # Cleanup
    vec_env.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)

