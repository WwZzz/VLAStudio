# 3. Evaluation in Simulation

This document explains how to evaluate a trained policy in a simulated environment using the `vlastudio evaluate` script.

## Core Script

`vlastudio evaluate` is the entry point for running evaluations. It orchestrates loading the policy, setting up the vectorized simulation environments, running the evaluation loop, and saving the results.

## Example Usage

Here is a concrete example of how to run the evaluation for the `sim_transfer_cube_scripted` task in the `aloha` benchmark. This command will run 10 episodes in parallel using 2 environments.

```bash
vlastudio evaluate --runtime current \
    -m ckpt/act_sim_transfer_cube_scripted_zscore_example \
    -e aloha \
    --num_rollout 10 \
    --num_envs 2 \
    --output_dir results/sim_eval_example \
    --env.task sim_transfer_cube_scripted
```

## Key Arguments

*   `-m, --model_name_or_path` (string):
    *   **Description**: Path to the saved model checkpoint directory.
    *   **Example**: `ckpt/act_sim_transfer_cube_scripted_zscore_example`

*   `-e, --env` (string):
    *   **Description**: The name of the environment configuration YAML file (located in `configs/env`). This file defines the benchmark, task, and simulation parameters.
    *   **Example**: `aloha` (refers to `configs/env/aloha.yaml`)

*   `--num_rollout` (int):
    *   **Description**: The total number of evaluation episodes to run.
    *   **Default**: `4`
    *   **Example**: `100`

*   `--num_envs` (int):
    *   **Description**: The number of parallel simulation environments to run. Using multiple environments can significantly speed up evaluation.
    *   **Default**: `2`
    *   **Example**: `8`

*   `-o, --output_dir` (string):
    *   **Description**: Directory where evaluation results (a `.json` file) and videos will be saved.
    *   **Default**: `results/dp_aloha...` (a default timestamped directory)
    *   **Example**: `results/my_eval_run`

*   `--use_spawn` (flag):
    *   **Description**: Use the 'spawn' method for multiprocessing instead of 'fork'. This can prevent certain deadlocks, especially when using CUDA.
    *   **Usage**: `--use_spawn`

## Overriding Configuration

You can override parameters in the environment configuration file directly from the command line by prefixing them with `--env.`.

For example, the `aloha.yaml` file specifies a default task. To evaluate on a different task like `sim_insertion`, you can do the following:

```bash
vlastudio evaluate --runtime current \
    -m <model_path> \
    -e aloha \
    --env.task sim_insertion
```

## Output Structure

The results are saved to the `--output_dir`. The structure looks like this:

```
<output_dir>/
└── <env_name>/
    ├── <task_name>.json
    └── video/
        ├── <task_name>_roll0_2.mp4
        └── <task_name>_roll2_4.mp4
```
*   **`<task_name>.json`**: Contains the final evaluation metrics, such as `total_success`, `total`, `success_rate`, and `horizon`.
*   **`video/`**: Contains the recorded videos of the evaluation rollouts.
