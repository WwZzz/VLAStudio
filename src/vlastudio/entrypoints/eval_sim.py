from vlastudio import configs  # Must be first to suppress TensorFlow logs
import os
import json
import importlib
import imageio
import multiprocessing as mp
import numpy as np
from loguru import logger
from vlastudio.data_utils.utils import set_seed
from tqdm import tqdm
from vlastudio.benchmark.utils import evaluate as default_evaluate, SequentialVectorEnv
from vlastudio.deploy.action_manager import load_action_manager
from vlastudio.deploy.inference import start_inference_process, stop_inference_process

def env_fn(env_config, env_handler):
    def create_env():
        return env_handler(env_config)
    return create_env

def load_env_module(env_cfg):
    """Resolve an environment definition to its create_env factory + evaluate fn.

    ``env_cfg.type`` may be any reference ``vlastudio.extensions.resolve``
    understands: a packaged dotted path (``pkg.mod.EnvClass`` /
    ``pkg.mod:EnvClass``), a local file (``/path/env.py:EnvClass``), or a plugin
    (``@env/name``). It may resolve to a module exposing ``create_env`` or
    directly to a class / callable used as the factory.
    """
    import inspect
    import sys
    from types import SimpleNamespace
    from vlastudio.extensions import resolve

    target = resolve(env_cfg.type, kind="env")
    if inspect.ismodule(target):
        create_env_fn = getattr(target, "create_env", None)
        module = target
        env_name = target.__name__.split(".")[-1]
    else:
        create_env_fn = target
        module = sys.modules.get(getattr(target, "__module__", ""))
        env_name = (getattr(target, "__module__", "") or "env").split(".")[-1]

    if not callable(create_env_fn):
        raise AttributeError(
            f"env type '{env_cfg.type}' did not resolve to a callable or a module "
            f"exposing 'create_env'"
        )

    if module is not None and hasattr(module, "evaluate"):
        evaluate = module.evaluate
        logger.info(f"Using environment-specific evaluate function from {env_name}")
    else:
        evaluate = default_evaluate
        logger.info("Using default evaluate function")

    env_module = SimpleNamespace(create_env=create_env_fn, evaluate=evaluate, __name__=env_name)
    return env_module, env_name, evaluate


def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model')
    parser.add_argument('-o', '--output_dir', type=str, default='results/dp_aloha_transer-official-ema-freq50-dnoise10-aug',
                    help='Directory to save results')
    # Model arguments
    parser.add_argument('-s', '--seed', type=int, default=0,
                       help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    # Model loading - can be checkpoint path or server address
    parser.add_argument('-m', '--model_name_or_path', type=str, 
                       default='localhost:5000',
                       help='Path to model checkpoint OR server address (host:port) for remote policy server')
    parser.add_argument('--dataset_id', type=str, default='',
                       help='Dataset ID to use (only for local model loading, ignored for remote server)')
    # Simulator arguments
    parser.add_argument('-e', '--env', type=str, default='aloha',
                       help='Env config (name under configs/env or absolute path to yaml)')
    # task/max_timesteps come from env config; override via --env.task / --env.max_timesteps if needed
    parser.add_argument('--fps', type=int, default=50,
                       help='Frames per second')
    parser.add_argument('-n', '--num_rollout', type=int, default=4,
                       help='Number of rollouts')
    parser.add_argument('-bs', '--batch_size', type=int, default=2,
                       help='Number of parallel environments (batch size)')
    parser.add_argument('--use_spawn', action='store_true',
                       help='Use spawn method for multiprocessing')
    # Model parameters (will be loaded from checkpoint config if not provided)
    parser.add_argument('-ck', '--chunk_size', type=int, default=-1,
                       help='Actual chunk size for policy that will truncate each raw chunk')
    # Action manager config
    parser.add_argument('-am', '--action_manager', type=str, default=None,
                       help='Action manager name or YAML config path. '
                            'e.g., "basic", "truncated_conservative", or path to YAML. '
                            'Default: "basic" (synchronous execution).')
    # Parse arguments
    args, unknown = parser.parse_known_args()
    # keep unknown tokens for env overrides (e.g., --env.task)
    args._unknown = unknown
    return args

if __name__=='__main__':
    args = parse_param()
    args.is_training = False
    set_seed(args.seed)
    if args.use_spawn: mp.set_start_method('spawn', force=True)
    
    # Start inference process (runs for entire evaluation session)
    inference_ctx = start_inference_process(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dataset_id=args.dataset_id,
        chunk_size=args.chunk_size,
    )
    
    # Load action manager (default: "basic" → BasicActionManager, synchronous)
    am_name = args.action_manager if args.action_manager is not None else "basic"
    action_manager = load_action_manager(am_name)
    action_manager.set_inference_context(inference_ctx)
    logger.info(f"ActionManager: {action_manager.__class__.__name__} (config: {am_name})")

    # load env via YAML config
    from vlastudio.configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, '_unknown', []))
    env_cfg, env_cfg_path = cfg_loader.load_env(args.env)
    env_cfgs = env_cfg if isinstance(env_cfg, list) else [env_cfg]
    
    # Store results for all environments
    all_env_results = {}
    
    try:
        # Iterate through each environment configuration
        for env_idx, env_cfg in enumerate(env_cfgs):
            logger.info("="*60)
            logger.info(f"Evaluating environment {env_idx + 1}/{len(env_cfgs)}")
            logger.info("="*60)
            
            # sync derived values from env config
            if hasattr(env_cfg, 'task'): args.task = env_cfg.task
            if hasattr(env_cfg, 'max_timesteps'): args.max_timesteps = env_cfg.max_timesteps
            
            env_module, env_name, evaluate = load_env_module(env_cfg)
            
            all_eval_results = []
            
            # When batch_size=0, run sequentially without SubprocVectorEnv (useful for environments with multiprocessing issues)
            # Determine execution mode and batch configuration
            use_sequential = (args.batch_size == 0)
            if use_sequential:
                logger.info(f"Running in sequential mode (batch_size=0, no SubprocVectorEnv)")
                num_iters = args.num_rollout
                batch_size = 1
            else:
                logger.info(f"Running in parallel mode (batch_size={args.batch_size}, using SubprocVectorEnv)")
                batch_size = args.batch_size
                num_iters = (args.num_rollout + batch_size - 1) // batch_size  # Ceiling division
            
            # Unified evaluation loop
            for i in tqdm(range(num_iters), total=num_iters, desc=f"Env {env_idx+1} Rollouts"):
                # Calculate number of environments for this iteration
                if use_sequential:
                    num_envs = 1
                    rollout_start = i
                    rollout_end = i + 1
                else:
                    num_envs = min(batch_size, args.num_rollout - i * batch_size)
                    rollout_start = i * batch_size
                    rollout_end = rollout_start + num_envs
                
                # Initialize video recorder
                if args.output_dir != '':
                    video_dir = os.path.join(args.output_dir, env_name, 'video')
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"{args.task}_roll{rollout_start}_{rollout_end}.mp4")
                    video_writer = imageio.get_writer(video_path, fps=args.fps)
                else:
                    video_writer = None
                
                # Create environment(s)
                env_fns = [env_fn(env_cfg, env_module.create_env) for _ in range(num_envs)]
                if use_sequential:
                    env = SequentialVectorEnv(env_fns)
                else:
                    try:
                        from tianshou.env import SubprocVectorEnv
                    except ImportError as error:
                        raise ImportError(
                            "Parallel evaluation requires tianshou. Add it to the environment "
                            "runtime or use batch_size=0."
                        ) from error
                    env = SubprocVectorEnv(env_fns)
                
                # Save example batch only for the first rollout
                save_example_dir = None
                if i == 0 and args.output_dir != '':
                    save_example_dir = os.path.join(args.output_dir, env_name, 'example_data')
                
                # Run evaluation
                eval_result = evaluate(
                    args, env, action_manager,
                    inference_ctx=inference_ctx,
                    video_writer=video_writer,
                    save_example_dir=save_example_dir,
                )
                logger.info(eval_result)
                all_eval_results.append(eval_result)
                
                # Reset action_manager (also resets inference subprocess policy)
                action_manager.reset()
                
                # Close video writer to flush the video to disk
                if video_writer is not None:
                    video_writer.close()
            
            eval_result = {
                'total_success': sum(eri['total_success'] for eri in all_eval_results),
                'total': sum(eri['total'] for eri in all_eval_results),
                'horizon': sum([eri['horizon'] for eri in all_eval_results], []),
                'horizon_success': sum([eri['horizon_success']*eri['total_success'] for eri in all_eval_results])
            }
            eval_result['success_rate'] = 1.0*eval_result['total_success']/eval_result['total']    
            eval_result['horizon_success']/=max(eval_result['total_success'], 1)  # Avoid division by zero
            
            # Add environment-specific metrics (e.g., CALVIN subtasks)
            # Collect any additional metrics that all results have
            if all_eval_results:
                first_result = all_eval_results[0]
                for key in first_result.keys():
                    if key not in eval_result and key not in ['success', 'horizon']:
                        # For list metrics, concatenate
                        if isinstance(first_result[key], list):
                            eval_result[key] = sum([eri.get(key, []) for eri in all_eval_results], [])
                        # For scalar metrics that look like rates/averages, compute average
                        elif isinstance(first_result[key], (int, float)):
                            if 'rate' in key or 'avg' in key or 'calvin_success' in key:
                                # Weighted average by number of samples
                                total_weight = sum(eri['total'] for eri in all_eval_results)
                                eval_result[key] = sum(eri.get(key, 0) * eri['total'] for eri in all_eval_results) / max(total_weight, 1)
                            else:
                                # For counts, sum them
                                eval_result[key] = sum(eri.get(key, 0) for eri in all_eval_results)
            
            # Store results for this environment
            env_key = f"{env_name}_{args.task}" if hasattr(env_cfg, 'task') else f"{env_name}_env{env_idx}"
            all_env_results[env_key] = eval_result
            
            # save result
            if args.output_dir!='':
                env_res_dir = os.path.join(args.output_dir, env_name)
                os.makedirs(env_res_dir, exist_ok=True)
                env_res_file = os.path.join(env_res_dir, f'{args.task}.json')
                with open(env_res_file, 'w') as f:
                    json.dump(eval_result, f)
            
            logger.info(f"Environment {env_idx + 1} Results:")
            logger.info(f"  Success Rate: {eval_result['success_rate']:.2%}")
            logger.info(f"  Total Success: {eval_result['total_success']}/{eval_result['total']}")
            logger.info(f"  Avg Horizon (Success): {eval_result['horizon_success']:.2f}")
        
        # Print summary for all environments
        if len(env_cfgs) > 1:
            logger.info("="*60)
            logger.info("Summary of All Environments:")
            logger.info("="*60)
            for env_key, result in all_env_results.items():
                logger.info(f"{env_key}: Success Rate = {result['success_rate']:.2%} ({result['total_success']}/{result['total']})")
            
            # Calculate average success rate across all environments
            total_success_all = sum(result['total_success'] for result in all_env_results.values())
            total_all = sum(result['total'] for result in all_env_results.values())
            avg_success_rate_weighted = total_success_all / total_all if total_all > 0 else 0.0
            
            # Calculate simple average (unweighted)
            avg_success_rate_simple = sum(result['success_rate'] for result in all_env_results.values()) / len(all_env_results)
            
            # Calculate average horizon for successful episodes across all environments
            total_horizon_success_weighted = sum(
                result['horizon_success'] * result['total_success'] 
                for result in all_env_results.values()
            )
            avg_horizon_success = total_horizon_success_weighted / total_success_all if total_success_all > 0 else 0.0
            
            logger.info("="*60)
            logger.info("Overall Statistics:")
            logger.info(f"  Average Success Rate (Weighted): {avg_success_rate_weighted:.2%} ({total_success_all}/{total_all})")
            logger.info(f"  Average Success Rate (Simple): {avg_success_rate_simple:.2%}")
            logger.info(f"  Average Horizon (Success): {avg_horizon_success:.2f}")
            logger.info("="*60)
            
            # Prepare summary with overall statistics
            summary_data = {
                'individual_environments': all_env_results,
                'overall_statistics': {
                    'avg_success_rate_weighted': avg_success_rate_weighted,
                    'avg_success_rate_simple': avg_success_rate_simple,
                    'total_success': total_success_all,
                    'total': total_all,
                    'avg_horizon_success': avg_horizon_success,
                    'num_environments': len(all_env_results)
                }
            }
            
            # Save combined results
            if args.output_dir!='':
                summary_file = os.path.join(args.output_dir, 'all_envs_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
    
    finally:
        # Always cleanup inference process
        stop_inference_process(inference_ctx)
