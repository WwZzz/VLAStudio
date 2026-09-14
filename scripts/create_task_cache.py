#!/usr/bin/env python3
"""Build all missing policy/task dataset caches without starting training."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from vlastudio.data_utils.task_cache import TaskCacheManager
from vlastudio.data_utils.utils import set_seed
from vlastudio.policy.policy_loader import (
    get_policy_data_collator,
    get_policy_data_processor,
    load_policy_model_for_training,
)
from train import load_all_configs, parse_param


def main(args):
    args.is_training = True
    task_config, policy_config, training_args, config_paths = load_all_configs(args)
    set_seed(getattr(training_args, 'seed', 0))
    model_components = load_policy_model_for_training(
        config_paths['policy'], args, task_config
    )
    processor = get_policy_data_processor(
        config_paths['policy'], args, model_components
    )
    collator = get_policy_data_collator(
        config_paths['policy'], args, model_components
    )
    manager = TaskCacheManager(
        args=args,
        task_config=task_config,
        policy_config=policy_config,
        processor=processor,
        collator=collator,
    )
    datasets = manager.ensure()
    logger.info(
        f"Task cache ready: {len(datasets)} datasets, level={manager.cache_level}, "
        f"format={manager.cache_format}, "
        f"path={manager.task_dir}"
    )


if __name__ == '__main__':
    main(parse_param())
