import os
from pathlib import Path

def resolve_yaml(name_or_path: str, base_dir: str) -> str:
    """Resolve a YAML config by either absolute/relative path or by name within base_dir.
    
    Supports dot notation for subdirectories:
        - 'rlbench.rlbench_reach' -> '{base_dir}/rlbench/rlbench_reach.yaml'
        - 'subdir.nested.config' -> '{base_dir}/subdir/nested/config.yaml'
    
    Returns an existing file path or raises FileNotFoundError.
    """
    if not name_or_path:
        raise FileNotFoundError("Empty config name or path")

    p = Path(name_or_path)
    # If looks like a path (has .yaml/.yml suffix or contains path separators), treat as path
    if p.suffix.lower() == '.yaml' or p.suffix.lower() == '.yml' or any(sep in name_or_path for sep in ['/', '\\']):
        candidate = Path(name_or_path)
        if candidate.exists():
            return str(candidate)
        # If it's a path without extension, try adding .yaml
        if candidate.suffix == '' and candidate.with_suffix('.yaml').exists():
            return str(candidate.with_suffix('.yaml'))
        raise FileNotFoundError(f"Config not found: {name_or_path}")

    # Treat as name under base_dir
    base = Path(base_dir)
    
    # Support dot notation for subdirectories: 'rlbench.rlbench_reach' -> 'rlbench/rlbench_reach.yaml'
    if '.' in name_or_path:
        # Convert dots to path separators
        subpath = name_or_path.replace('.', os.sep)
        candidate = base / f"{subpath}.yaml"
        if candidate.exists():
            return str(candidate)
        # Also try without conversion (for backward compatibility with names containing dots)
        candidate_flat = base / f"{name_or_path}.yaml"
        if candidate_flat.exists():
            return str(candidate_flat)
        raise FileNotFoundError(f"Config '{name_or_path}' not found under {base_dir} (tried: {subpath}.yaml, {name_or_path}.yaml)")
    
    candidate = base / f"{name_or_path}.yaml"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Config '{name_or_path}' not found under {base_dir}")


def _set_nested(obj, keys, value):
    cur = obj
    for k in keys[:-1]:
        if isinstance(cur, dict):
            if k not in cur or not isinstance(cur[k], (dict,)):
                cur[k] = {}
            cur = cur[k]
        else:
            if not hasattr(cur, k) or not isinstance(getattr(cur, k), (dict,)):
                setattr(cur, k, {})
            cur = getattr(cur, k)
    last = keys[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def parse_overrides(unknown_args):
    """Parse command line overrides with support for arbitrary nesting depth.
    
    Supports patterns like:
    - --policy.camera_names ["primary", "wrist"]
    - --policy.model_args.backbone resnet50
    - --training.optimizer.lr_scheduler.type cosine
    - --task.env.simulation.physics.timestep 0.01
    
    Args:
        unknown_args: List of unknown command line arguments
        
    Returns:
        Dict with structure: {category: {nested_path: value, ...}, ...}
        where nested_path can be arbitrarily deep (e.g., "model_args.backbone.layers")
    """
    overrides = {
        'task': {},
        'training': {},
        'policy': {},
        'teleop': {},
        'robot': {},
        'env': {},
        'action_manager': {},
    }
    supported_roots = tuple(overrides.keys())
    
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith('--'):
            i += 1
            continue
            
        key = token[2:]
        value = None
        
        # Handle --key=value syntax
        if '=' in key:
            key, value = key.split('=', 1)
        else:
            # Handle --key value syntax
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith('--'):
                value = unknown_args[i+1]
                i += 1
        
        # Check if key starts with any of our supported root categories
        key_parts = key.split('.')
        if len(key_parts) >= 2 and key_parts[0] in supported_roots:
            root = key_parts[0]
            # Join all remaining parts as the nested path
            # This supports arbitrary depth: policy.model_args.backbone.layers
            nested_path = '.'.join(key_parts[1:])
            overrides[root][nested_path] = value
        
        i += 1
    return overrides


def convert_yaml_string_types(config_dict):
    """
    Recursively convert string values in a config dict to appropriate types.
    Handles scientific notation like '1e-8', numbers like '123', bools like 'true'.
    
    This is needed because yaml.safe_load() sometimes parses scientific notation
    as strings (e.g., '1e-8' instead of 1e-08).
    
    Args:
        config_dict: Configuration dictionary (modified in-place)
    """
    def convert_value(value):
        """Convert a single value to appropriate type."""
        if not isinstance(value, str):
            return value
        
        # Check for boolean
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        
        # Check for integer
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        
        # Check for float (including scientific notation)
        try:
            return float(value)
        except ValueError:
            # Keep as string if conversion fails
            return value
    
    def recursive_convert(obj):
        """Recursively convert all string values in nested dict/list structures."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    obj[key] = convert_value(value)
                elif isinstance(value, (dict, list)):
                    recursive_convert(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = convert_value(item)
                elif isinstance(item, (dict, list)):
                    recursive_convert(item)
    
    recursive_convert(config_dict)
    return config_dict


def apply_overrides_to_mapping(mapping_obj, flat_overrides, caster):
    for dotted, raw in flat_overrides.items():
        if raw is None:
            continue
        try:
            val = caster(raw)
        except Exception:
            val = raw
        keys = dotted.split('.')
        _set_nested(mapping_obj, keys, val)


def apply_overrides_to_object(obj, flat_overrides, caster):
    for dotted, raw in flat_overrides.items():
        if raw is None:
            continue
        try:
            val = caster(raw)
        except Exception:
            val = raw
        keys = dotted.split('.')
        _set_nested(obj, keys, val)


