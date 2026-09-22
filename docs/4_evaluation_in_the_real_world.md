# 4. Evaluation in the Real World

This guide covers how to deploy and evaluate a trained policy on a physical robot using the `vlastudio infer` script.

## ⚠️ Safety First!

**Warning**: Running policies on a real robot can be dangerous. Real-world hardware can behave unexpectedly.
*   **Always be prepared to stop the robot.** Keep the emergency stop button within reach.
*   **Clear the workspace.** Ensure the robot's operating area is free of any obstacles or personnel.
*   **Start with low speeds.** If possible, test at a reduced speed before running at full speed.

## System Architecture

The current `vlastudio infer` script uses a multi-process, shared-memory architecture:

1.  **Device Processes**: Robot and camera devices run in their own subprocesses and publish data into shared memory.
2.  **Inference Worker**: A dedicated subprocess reads device shared memory directly, synchronizes observations, runs policy inference, and writes action chunks into its own shared-memory channel.
3.  **Main Control Loop**: Runs at the robot's required control frequency (`--publish_rate`). It continuously queries the **Action Manager** for the next action and sends it to the robot hardware through `policy_control_shm`.

## Example Usage

This example shows how to run an evaluation on a real robot with the current CLI.

```bash
vlastudio infer --runtime current \
    --model_name_or_path ckpt/act_sim_transfer_cube_scripted_zscore_example \
    -r agilex_aloha \
    --publish_rate 50 \
    --sensing_rate 25 \
    --action_manager basic
```

## Key Arguments

*   `--model_name_or_path` (string):
    *   **Description**: Path to the model checkpoint *or* a server address (`host:port`) if using a remote Policy Server.
    *   **Example (local)**: `ckpt/act_sim_transfer_cube_scripted_zscore_example`
    *   **Example (remote)**: `192.168.1.101:5000`

*   `-r` / `--robot` (string), same as `vlastudio collect`:
    *   **Description**: Robot configuration name under `configs/robot/` or path to a YAML. Defines hardware, cameras, etc.
    *   **Example**: `agilex_aloha` → `configs/robot/agilex_aloha.yaml`
    *   **Alias**: `--robot_config` / `--robot-config` still accepted for older scripts.

*   `--task` (string):
    *   **Description**: The name of the task configuration YAML file from `configs/task/`. This defines the datasets, normalization, and policy settings used for training, which are needed to load the model correctly.
    *   **Example**: `agilex_transfer_cube`

*   `--publish_rate` (int):
    *   **Description**: The frequency (Hz) at which the main control loop sends action commands to the robot. This should match the robot's expected control rate.
    *   **Default**: `25`

*   `--sensing_rate` (int):
    *   **Description**: The frequency (Hz) at which the sensing thread polls the robot for new observations.
    *   **Default**: `20`

*   `--action_manager` (string):
    *   **Description**: The name of the Action Manager class to use. See the Action Manager documentation for more details.
    *   **Default**: `basic`

## Pre-flight Checklist

1.  ✅ **Robot On**: The robot is powered on and initialized.
2.  ✅ **Network Connection**: Your machine can communicate with the robot (e.g., via ROS, Ethernet).
3.  ✅ **Drivers Running**: The robot's low-level control software/drivers are running.
4.  ✅ **Correct Configs**: The `-r` / `--robot` and `--task` files accurately reflect your setup.
5.  ✅ **Safety**: The workspace is clear and you are ready to stop the robot if needed.
