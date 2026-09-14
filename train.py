import configs 
import os
import argparse
import json
from loguru import logger
import policy.utils as ml_utils
from data_utils.utils import set_seed, load_data, save_example_data
from data_utils.data_loader import get_dataloader
from configs.loader import ConfigLoader
from policy.policy_loader import (
    get_policy_data_processor,
    get_policy_data_collator,
    get_policy_trainer_class,
    load_policy_model_for_training,
)
from policy.trainer import BaseTrainer


def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Train a policy model')
    
    # Essential arguments
    parser.add_argument('-p', '--policy', type=str, default='act',
                       help='Policy config (name under configs/policy or absolute path to yaml)')
    parser.add_argument('-t', '--task', type=str, default='sim_transfer_cube_scripted',
                       help='Task config (name under configs/task or absolute path to yaml)')
    parser.add_argument('-c', '--training_config', type=str, default='default',
                       help='Training config (name under configs/training or absolute path to yaml)')
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/training_output',
                       help='Output directory for checkpoints')
    parser.add_argument('--eval_ratio', type=float, default=0.0,
                       help='Ratio of training data to use for evaluation. Default to 0.0 (use first dataset as eval if multiple provided)')
    
    # Parse arguments (allow unknown for dotted overrides)
    args, unknown = parser.parse_known_args()
    
    # Store unknown args for later use
    setattr(args, 'unknown_args', unknown)
    return args

def load_all_configs(args):
    """
    Load all configurations in one place.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (task_config, policy_config, training_args, config_paths)
    """
    # Create config loader once
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'unknown_args', []))
    
    # Load all configurations
    task_config, task_cfg_path = cfg_loader.load_task(args.task)
    policy_config, policy_cfg_path = cfg_loader.load_policy(args.policy)
    training_config, training_args, training_cfg_path = cfg_loader.load_training(args.training_config, hyper_args=args)
    
    # Merge all parameters
    ConfigLoader.merge_all_parameters(task_config, policy_config, training_config, args)
    
    config_paths = {
        'task': task_cfg_path,
        'policy': policy_cfg_path,
        'training': training_cfg_path
    }
    
    return task_config, policy_config, training_args, config_paths

def main(args):
    """
    Main training function for the VLA (Vision-Language-Action) model.

    Args:
        args (HyperArguments): Training hyperparameters and settings

    Returns:
        None. The trained model and statistics are saved to the output directory
        specified in training_args.
    """
    args.is_training = True
    
    # Load all configurations in one place
    task_config, policy_config, training_args, config_paths = load_all_configs(args)
    from data_utils.task_cache import is_task_cache_enabled

    cache_enabled = is_task_cache_enabled(task_config)
    
    # Set random seed
    seed = getattr(training_args, 'seed', 0)
    set_seed(seed)
    logger.info(f"🌱 Set global seed to: {seed} for reproducibility")
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    all_ckpts = [os.path.join(training_args.output_dir, ckpt_name) for ckpt_name in os.listdir(training_args.output_dir) if ckpt_name.startswith('checkpoint-') and os.path.isdir(os.path.join(training_args.output_dir, ckpt_name))]
    if len(all_ckpts)==0: training_args.resume_from_checkpoint = None
    
    # Save policy metadata to output dir
    metadata_path = os.path.join(training_args.output_dir, 'policy_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
                'policy_module': policy_config.get('module_path') or policy_config['type'],
                'policy_name': policy_config['name'],
                **({'runtime': json.loads(os.environ['VLASTUDIO_RUNTIME_JSON'])} if os.environ.get('VLASTUDIO_RUNTIME_JSON') else {}),
            }, f, indent=2)
    
    # Load model 
    logger.info(f"Loading policy config: {config_paths['policy']}")
    model_components = load_policy_model_for_training(config_paths['policy'], args, task_config)
    model = model_components['model']
    config = model_components.get('config', None)
    if config:
        logger.info(f"Loaded config from YAML: {type(config).__name__}") 
    ml_utils.print_model_trainable_information(model)
    
    # Resolve the policy pipeline before loading data. Cache construction runs
    # through the configured boundary; cache hits always bypass the processor,
    # while processor-level caches retain the original policy collator.
    data_processor = get_policy_data_processor(config_paths['policy'], args, model_components)
    data_collator = get_policy_data_collator(config_paths['policy'], args, model_components)
    if cache_enabled:
        from data_utils.task_cache import load_task_cache

        data_dict, data_collator, cache_manager = load_task_cache(
            args=args,
            task_config=task_config,
            policy_config=policy_config,
            processor=data_processor,
            collator=data_collator,
            output_dir=training_args.output_dir,
        )
        data_processor = None
        logger.info(
            f"Using {cache_manager.cache_level}-level task cache "
            f"({cache_manager.cache_format}) from {cache_manager.task_dir}"
        )
    else:
        data_dict = load_data(args, task_config)

    train_data, val_data = data_dict['train'], data_dict['eval']

    if not cache_enabled:
        # Cached samples are already policy batches, so the raw-data visualizer is
        # intentionally only used on the uncached path.
        logger.info("="*80)
        logger.info(f"Saving example data to {training_args.output_dir}...")
        logger.info("="*80)
        save_example_data(train_data, training_args.output_dir)
        logger.info("="*80)

    # Create data loader with either the policy pipeline or generic cache pipeline.
    train_loader, eval_loader = get_dataloader(train_data, val_data, data_processor, data_collator, args) 
    # assert val_data is not None, "Validation data is required for training"
    # Get Trainer
    train_class = get_policy_trainer_class(config_paths['policy']) or BaseTrainer
    trainer = train_class(
        args=training_args,
        model=model,
        tokenizer=model_components.get('tokenizer', None),
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # Save model
    if trainer.is_world_process_zero():
        trainer.save_state()
        trainer.save_model(training_args.output_dir)

if __name__ == '__main__':
    args = parse_param()
    main(args)
