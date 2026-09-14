"""
CALVIN Evaluation Functions

This module provides evaluation functions specifically designed for CALVIN's
multi-step, long-horizon task evaluation protocol.
"""

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from typing import Optional, Dict, Any
import os
from pathlib import Path


def count_success(results):
    """
    Count success rates for each position in the sequence.
    
    Args:
        results: List of integers, where each integer is the number of subtasks completed
        
    Returns:
        List of 5 floats representing success rate at each position (1/5, 2/5, 3/5, 4/5, 5/5)
    """
    count = [0] * 5
    for result in results:
        for i in range(result):
            count[i] += 1
    return [c / len(results) for c in count]


def evaluate(args, action_manager, env, inference_ctx=None, video_writer=None, save_example_dir=None):
    """
    CALVIN-specific evaluate function.
    
    This function properly evaluates CALVIN sequences by tracking subtask completion
    rather than just binary success/failure.
    
    Args:
        args: Configuration args
        action_manager: ActionManager with InferenceContext set.
        env: Vector environment (SequentialVectorEnv or SubprocVectorEnv)
        inference_ctx: InferenceContext for writing obs to SHM (sim mode).
        video_writer: Optional video writer
        save_example_dir: Optional directory to save example data
        
    Returns:
        Dict with CALVIN-specific metrics including subtask completion rates
    """
    from vlastudio.benchmark.utils import organize_obs
    import imageio
    
    num_envs = len(env)
    video_frames = [[] for _ in range(num_envs)]
    horizons = np.ones(num_envs) * args.max_timesteps
    
    # Track subtask completion for each environment
    subtasks_completed = np.zeros(num_envs, dtype=np.int32)
    success = np.zeros(num_envs, dtype=np.bool_)
    done_flags = np.zeros(num_envs, dtype=np.bool_)
    
    with torch.inference_mode():
        obs = env.reset()
        obs = organize_obs(obs)
        
        # Save first observation and action for debugging
        first_obs_saved = False
        
        for t in range(args.max_timesteps):
            # Record video frames
            if video_writer is not None:
                frames = obs['image']
                if len(frames.shape) == 5:
                    frames = frames[:, 0]
                frames = frames.transpose(0, 2, 3, 1)
                for env_i in range(num_envs):
                    if not done_flags[env_i]:
                        video_frames[env_i].append(frames[env_i])
            
            # Write obs to SHM for inference worker (sim mode)
            if inference_ctx is not None:
                inference_ctx.update_obs(obs, t)
            
            # Get action from action_manager
            act = action_manager.select_action()
            
            # Save first obs and action
            if not first_obs_saved and save_example_dir is not None:
                from vlastudio.benchmark.utils import _save_example_batch
                _save_example_batch(obs, act, save_example_dir)
                first_obs_saved = True
            
            # Step environment
            obs, reward, done, info = env.step(act, id=None)
            obs = organize_obs(obs)
            
            # Update completion tracking for each environment
            for env_i in range(num_envs):
                if not done_flags[env_i]:
                    env_info = info[env_i] if isinstance(info, list) else info
                    
                    # Update subtasks completed
                    subtasks_completed[env_i] = env_info.get('subtasks_completed', 0)
                    
                    # Check if sequence is done
                    if done[env_i]:
                        done_flags[env_i] = True
                        horizons[env_i] = t
                        # In CALVIN, success means completing at least 1 subtask
                        success[env_i] = subtasks_completed[env_i] > 0
            
            # Stop if all environments are done
            if done_flags.all():
                break
        
        # For any environments that didn't finish
        for env_i in range(num_envs):
            if not done_flags[env_i]:
                horizons[env_i] = args.max_timesteps
                success[env_i] = subtasks_completed[env_i] > 0
    
    env.close()
    
    # Save video
    if video_writer is not None:
        for env_i in range(num_envs):
            for frame in video_frames[env_i]:
                video_writer.append_data(frame)
    
    # Compute CALVIN-specific metrics
    total_successes = int(success.sum().item())
    total = num_envs
    success_rate = 1.0 * total_successes / total
    
    # Compute subtask completion rates (1/5, 2/5, 3/5, 4/5, 5/5)
    calvin_rates = {}
    for i in range(1, 6):
        calvin_rates[f'calvin_success_{i}'] = (subtasks_completed >= i).sum() / total
    
    # Average number of subtasks completed
    avg_subtasks = subtasks_completed.mean()
    
    result = {
        'success': success.tolist(),
        'total_success': total_successes,
        'total': total,
        'success_rate': success_rate,  # At least 1 subtask completed
        'horizon': horizons.tolist(),
        'horizon_success': (success * horizons).sum() / max(total_successes, 1),
        'subtasks_completed': subtasks_completed.tolist(),
        'avg_subtasks_completed': float(avg_subtasks),
        **calvin_rates,  # Add CALVIN-specific metrics
    }
    
    return result


def evaluate_calvin(
    args,
    policy,
    env_class,
    num_sequences: int = 1000,
    max_steps_per_subtask: int = 360,
    save_videos: bool = False,
    video_dir: Optional[str] = None,
    num_videos: int = 10,
):
    """
    Evaluate a policy on CALVIN benchmark.
    
    This function evaluates across multiple sequences, where each sequence
    contains up to 5 subtasks. The key metric is how many subtasks can be
    completed in a row.
    
    Args:
        args: Configuration args containing task info
        policy: Policy wrapped in MetaPolicy
        env_class: CalvinEnv class or create_env function
        num_sequences: Number of sequences to evaluate
        max_steps_per_subtask: Maximum steps per subtask (default: 360)
        save_videos: Whether to save trajectory videos
        video_dir: Directory to save videos
        num_videos: Number of sequences to save as videos
        
    Returns:
        Dict with evaluation results including:
            - results: List of completed subtasks per sequence
            - success_rates: Success rate at each position (1/5, 2/5, etc.)
            - avg_sequence_length: Average number of completed subtasks
            - total_sequences: Total number of sequences evaluated
    """
    logger.info(f"Starting CALVIN evaluation on {num_sequences} sequences")
    logger.info(f"Task: {args.task}, Max steps per subtask: {max_steps_per_subtask}")
    
    # Prepare video saving if requested
    video_frames = []
    if save_videos:
        if video_dir is None:
            video_dir = "./calvin_videos"
        os.makedirs(video_dir, exist_ok=True)
        logger.info(f"Saving videos to {video_dir}")
    
    results = []
    
    # Create a progress bar
    pbar = tqdm(range(num_sequences), desc="Evaluating CALVIN sequences")
    
    for seq_idx in pbar:
        # Create a new environment for this sequence
        # Each env instance corresponds to one sequence
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            'task': args.task,
            'show_gui': False,
            'num_sequences': num_sequences,
            'sequence_idx': seq_idx,
        })
        
        env = env_class(config)
        
        # Reset policy for new sequence
        if hasattr(policy, 'reset'):
            policy.reset()
        
        # Evaluate this sequence
        if save_videos and seq_idx < num_videos:
            # TODO: Implement video recording
            subtasks_completed = evaluate_calvin_sequence(env, policy, max_steps_per_subtask)
        else:
            subtasks_completed = evaluate_calvin_sequence(env, policy, max_steps_per_subtask)
        
        results.append(subtasks_completed)
        
        # Update progress bar with current success rates
        success_rates = count_success(results)
        avg_length = np.mean(results)
        pbar.set_postfix({
            '1/5': f'{success_rates[0]:.2%}',
            '2/5': f'{success_rates[1]:.2%}',
            '3/5': f'{success_rates[2]:.2%}',
            '4/5': f'{success_rates[3]:.2%}',
            '5/5': f'{success_rates[4]:.2%}',
            'avg': f'{avg_length:.2f}',
        })
        
        # Clean up
        env.close()
    
    # Compute final metrics
    success_rates = count_success(results)
    avg_sequence_length = np.mean(results)
    
    # Log final results
    logger.info("=" * 60)
    logger.info("CALVIN Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Task: {args.task}")
    logger.info(f"Total sequences: {num_sequences}")
    logger.info(f"Average sequence length: {avg_sequence_length:.3f}")
    logger.info("")
    logger.info("Success rates:")
    for i, rate in enumerate(success_rates, 1):
        logger.info(f"  {i}/5: {rate:.2%} ({int(rate * num_sequences)}/{num_sequences})")
    logger.info("=" * 60)
    
    return {
        'results': results,
        'success_rates': success_rates,
        'avg_sequence_length': avg_sequence_length,
        'total_sequences': num_sequences,
        'task': args.task,
    }


def evaluate_calvin_parallel(
    args,
    policy,
    env_class,
    num_sequences: int = 1000,
    num_parallel_envs: int = 1,
    max_steps_per_subtask: int = 360,
):
    """
    Evaluate CALVIN with parallel environments.
    
    Note: Currently CALVIN evaluation is sequential because each sequence
    requires specific initialization. This function is for future extension.
    
    Args:
        args: Configuration args
        policy: Policy wrapped in MetaPolicy
        env_class: CalvinEnv class
        num_sequences: Number of sequences to evaluate
        num_parallel_envs: Number of parallel environments (currently must be 1)
        max_steps_per_subtask: Maximum steps per subtask
        
    Returns:
        Dict with evaluation results
    """
    if num_parallel_envs > 1:
        logger.warning(
            "Parallel evaluation is not yet supported for CALVIN. "
            "Falling back to sequential evaluation."
        )
    
    return evaluate_calvin(
        args=args,
        policy=policy,
        env_class=env_class,
        num_sequences=num_sequences,
        max_steps_per_subtask=max_steps_per_subtask,
    )

