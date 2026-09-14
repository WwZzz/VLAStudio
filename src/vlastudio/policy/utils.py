from loguru import logger
from vlastudio.deploy.comm import create_client, is_server_address, is_http_address
import numpy as np
import re


class DummyPolicy:
    """
    Dummy policy for testing environment pipeline.
    
    Supports two modes:
    - zero: Returns all-zero actions
    - random: Returns random actions
    
    Usage:
        __dummy-zero7: 7-dim zero actions, chunk_size=1 (default)
        __dummy-random7: 7-dim random actions, chunk_size=1
        __dummy-zero7-chunk50: 7-dim zero actions, chunk_size=50
        __dummy-random14-chunk16: 14-dim random actions, chunk_size=16
        __dummy-14: 14-dim zero actions, chunk_size=1
        __dummy-zero: 7-dim zero actions, chunk_size=1 (defaults)
    """
    
    def __init__(self, action_dim=7, mode='zero', chunk_size=1):
        """
        Args:
            action_dim: Dimension of action space
            mode: 'zero' or 'random'
            chunk_size: Number of actions to return (default: 1)
        """
        self.action_dim = action_dim
        self.mode = mode
        self.chunk_size = chunk_size
        logger.info(f"🎲 Created DummyPolicy: mode={mode}, action_dim={action_dim}, chunk_size={chunk_size}")
    
    def select_action(self, obs, *args, **kwargs):
        """
        Return dummy actions with shape (batch_size, chunk_size, action_dim).
        Compatible with MetaPolicy interface.
        
        Args:
            obs: Observation dict or any input (used to infer batch_size)
            *args, **kwargs: Accept any additional arguments for compatibility
        
        Returns:
            np.ndarray: Shape (batch_size, chunk_size, action_dim) if chunk_size > 1
                       or (batch_size, action_dim) if chunk_size == 1
        """
        # Infer batch_size from observation
        batch_size = 1
        if isinstance(obs, dict):
            # Try to get batch_size from common keys
            for key in ['state', 'image', 'lang_tokens']:
                if key in obs and obs[key] is not None:
                    if hasattr(obs[key], 'shape') and len(obs[key].shape) > 0:
                        batch_size = obs[key].shape[0]
                        break
        
        # Generate actions based on mode
        if self.mode == 'zero':
            if self.chunk_size == 1:
                actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
            else:
                actions = np.zeros((batch_size, self.chunk_size, self.action_dim), dtype=np.float32)
        elif self.mode == 'random':
            if self.chunk_size == 1:
                actions = np.random.randn(batch_size, self.action_dim).astype(np.float32) * 0.1
            else:
                actions = np.random.randn(batch_size, self.chunk_size, self.action_dim).astype(np.float32) * 0.1
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return actions
    
    def eval(self):
        """Dummy eval method for compatibility."""
        pass
    
    def train(self):
        """Dummy train method for compatibility."""
        pass


def parse_dummy_policy_config(model_name_or_path):
    """
    Parse dummy policy configuration from model_name_or_path.
    
    Format:
        __dummy-zero7: mode=zero, dim=7, chunk_size=1
        __dummy-random14: mode=random, dim=14, chunk_size=1
        __dummy-zero7-chunk50: mode=zero, dim=7, chunk_size=50
        __dummy-random14-chunk16: mode=random, dim=14, chunk_size=16
        __dummy-7: mode=zero (default), dim=7, chunk_size=1
        __dummy-7-chunk10: mode=zero, dim=7, chunk_size=10
        __dummy-zero: mode=zero, dim=7 (default), chunk_size=1
        __dummy-zero-chunk20: mode=zero, dim=7, chunk_size=20
    
    Returns:
        (mode, action_dim, chunk_size): tuple of (str, int, int)
    """
    # Default values
    default_mode = 'zero'
    default_dim = 7
    default_chunk = 1
    
    # Remove __dummy- prefix
    config_str = model_name_or_path[len('__dummy-'):]
    
    # Parse mode, dimension, and chunk_size
    mode = default_mode
    action_dim = default_dim
    chunk_size = default_chunk
    
    # Try to match patterns with chunk size
    # Pattern 1: __dummy-{mode}{dim}-chunk{size}, e.g., __dummy-zero7-chunk50
    match = re.match(r'^(zero|random)(\d+)-chunk(\d+)$', config_str)
    if match:
        mode = match.group(1)
        action_dim = int(match.group(2))
        chunk_size = int(match.group(3))
        return mode, action_dim, chunk_size
    
    # Pattern 2: __dummy-{dim}-chunk{size}, e.g., __dummy-7-chunk50
    match = re.match(r'^(\d+)-chunk(\d+)$', config_str)
    if match:
        action_dim = int(match.group(1))
        chunk_size = int(match.group(2))
        return default_mode, action_dim, chunk_size
    
    # Pattern 3: __dummy-{mode}-chunk{size}, e.g., __dummy-zero-chunk50
    match = re.match(r'^(zero|random)-chunk(\d+)$', config_str)
    if match:
        mode = match.group(1)
        chunk_size = int(match.group(2))
        return mode, default_dim, chunk_size
    
    # Pattern 4: __dummy-{mode}{dim}, e.g., __dummy-zero7, __dummy-random14
    match = re.match(r'^(zero|random)(\d+)$', config_str)
    if match:
        mode = match.group(1)
        action_dim = int(match.group(2))
        return mode, action_dim, default_chunk
    
    # Pattern 4b: __dummy-{dim}{mode}, e.g., __dummy-16random, __dummy-7zero
    match = re.match(r'^(\d+)(zero|random)$', config_str)
    if match:
        action_dim = int(match.group(1))
        mode = match.group(2)
        return mode, action_dim, default_chunk
    
    # Pattern 5: __dummy-{dim}, e.g., __dummy-7, __dummy-14
    match = re.match(r'^(\d+)$', config_str)
    if match:
        action_dim = int(match.group(1))
        return default_mode, action_dim, default_chunk
    
    # Pattern 6: __dummy-{mode}, e.g., __dummy-zero, __dummy-random
    match = re.match(r'^(zero|random)$', config_str)
    if match:
        mode = match.group(1)
        return mode, default_dim, default_chunk
    
    # If nothing matches, return defaults
    logger.warning(f"Could not parse dummy config '{config_str}', using defaults: mode={default_mode}, dim={default_dim}, chunk_size={default_chunk}")
    return default_mode, default_dim, default_chunk


def load_policy(args):
    # Check if model_name_or_path is a server address or local checkpoint
    if is_server_address(args.model_name_or_path):
        logger.info("="*60)
        logger.info("🤖 Remote Policy Evaluation")
        logger.info("="*60)
        address = args.model_name_or_path
        if is_http_address(address):
            logger.info(f"🌐 Using remote FastAPI policy server: {address}")
        else:
            logger.info(f"🌐 Using remote TCP policy server: {address}")
        
        # Create remote policy client (auto-detect TCP vs HTTP)
        policy = create_client(
            address=address,
            timeout_s=3600.0,  # Set timeout to 3600 seconds for long-running inference
        )
        
        # Set dummy values for compatibility
        # For real robot eval, these will be updated after policy is created
        if not hasattr(args, 'ctrl_space'):
            args.ctrl_space = policy.ctrl_space
            args.ctrl_type = policy.ctrl_type
    
    elif args.model_name_or_path.startswith('__dummy-'):
        # Dummy policy mode for testing pipeline
        logger.info("="*60)
        logger.info("🎲 Dummy Policy Mode (Testing)")
        logger.info("="*60)
        
        # Parse dummy policy configuration
        mode, action_dim, chunk_size = parse_dummy_policy_config(args.model_name_or_path)
        logger.info(f"📋 Dummy Policy Config: mode={mode}, action_dim={action_dim}, chunk_size={chunk_size}")
        
        # Create Identity normalizers (no normalization)
        from vlastudio.data_utils.normalize import Identity
        from vlastudio.benchmark.base import MetaPolicy
        
        identity_normalizer = Identity()
        logger.info("✓ Using Identity normalizers (no normalization)")
        
        # Create dummy policy with chunk_size
        dummy_model = DummyPolicy(action_dim=action_dim, mode=mode, chunk_size=chunk_size)
        
        # Set ctrl_space and ctrl_type if not already set
        if not hasattr(args, 'ctrl_space'):
            args.ctrl_space = 'ee'
        if not hasattr(args, 'ctrl_type'):
            args.ctrl_type = 'delta'
        
        logger.info(f"✓ Control space: {args.ctrl_space}, Control type: {args.ctrl_type}")
        
        # Set chunk_size in args for MetaPolicy (will override if specified in config)
        if not hasattr(args, 'chunk_size') or args.chunk_size is None:
            args.chunk_size = chunk_size
        
        # Wrap in MetaPolicy
        policy = MetaPolicy(
            policy=dummy_model,
            action_normalizer=identity_normalizer,
            state_normalizer=identity_normalizer,
            ctrl_space=args.ctrl_space,
            ctrl_type=args.ctrl_type,
        )
        
        logger.info("✓ Dummy policy wrapped in MetaPolicy")
        logger.info("="*60)
        
    else:
        # Local model mode (fallback to original behavior)
        logger.info("="*60)
        logger.info("🤖 Local Policy Evaluation")
        logger.info("="*60)
        # Load normalizers and model as before
        from vlastudio.data_utils.normalize import load_normalizers
        from vlastudio.benchmark.base import MetaPolicy
        
        normalizers, ctrl_space, ctrl_type = load_normalizers(args)
        args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type
        
        # Load policy directly from checkpoint
        logger.info(f"Loading model from checkpoint: {args.model_name_or_path}")
        # Fallback to direct checkpoint loading
        from vlastudio.policy.direct_loader import load_model_from_checkpoint
        if not hasattr(args, 'is_training'):
            args.is_training = False
        model_components = load_model_from_checkpoint(args.model_name_or_path, args)
        model = model_components['model']
        config = model_components.get('config', None)
        if config:
            logger.info(f"Loaded config from checkpoint: {type(config).__name__}")
        
        # Always wrap model in MetaPolicy
        policy = MetaPolicy(
            policy=model, 
            action_normalizer=normalizers['action'], 
            state_normalizer=normalizers['state'], 
            ctrl_space=ctrl_space, 
            ctrl_type=ctrl_type,
        )
    return policy

def print_model_trainable_information(model, rank0_print=None):
    if rank0_print is None:
        rank0_print = logger.info
    lora_para = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n)
    all_para = sum(p.numel() for n, p in model.named_parameters())
    train_para = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    frozen_para = all_para - train_para
    rank0_print(
        f"LoRA trainable / all trainable / total / frozen: "
        f"{lora_para / 1e6:.3f}M / {train_para / 1e6:.3f}M / {all_para / 1e6:.3f}M / {frozen_para / 1e6:.3f}M"
    )