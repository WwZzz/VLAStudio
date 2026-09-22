# 1. Overview

Welcome to **IL-Studio**, a comprehensive, modular, and easy-to-use framework designed for advanced Imitation Learning (IL) research.

## Mission

*   **Accelerate Research**: Provide a powerful and flexible platform that enables researchers to rapidly prototype, train, and evaluate imitation learning algorithms.
*   **Bridge Sim-to-Real**: Offer robust tools and standardized workflows to facilitate the transfer of policies from simulation to physical hardware.
*   **Promote Modularity**: Foster a modular architecture where components like policies, datasets, and environments can be easily swapped and customized.

## Core Concepts

*   **Unified Configuration (`configs/`)**: The entire framework is driven by YAML configuration files. This allows you to define everything from datasets and policy architectures to training regimes and robot hardware in a centralized, readable format.

*   **Policies (`policy/`)**: The "brain" of the agent. IL-Studio supports a wide range of policy architectures, from simple MLPs to complex transformer-based models like ACT and Diffusion Policy. All policies are wrapped in a `MetaPolicy`.

*   **MetaPolicy (`benchmark/base.py`)**: A crucial wrapper that standardizes the interaction with any raw policy. It handles common but critical tasks like:
    *   **Data Normalization**: Applying `ZScore`, `MinMax`, or `Percentile` normalization to state and action data.
    *   **Action Chunking**: Managing the temporal dimension of actions, where a policy might predict a sequence (chunk) of future actions at once.

*   **Datasets (`data_utils/datasets/`)**: Flexible data loaders designed to handle various robotics dataset formats, including RLDS, Robomimic HDF5, and custom formats.

*   **Benchmarks (`benchmark/`)**: Standardized simulation environments for reproducible evaluation. We provide wrappers for popular benchmarks like `robomimic`, `panda-gym`, and our custom `aloha` environment.

*   **Deployment (`deploy/`)**: The toolset for running policies outside of a simple training script. This includes:
    *   **Robot Interfaces (`deploy/robot/`)**: A hardware abstraction layer for controlling physical robots.
    *   **Teleoperation (`deploy/teleoperator/`)**: Tools for collecting expert demonstration data.
    *   **Policy Server (`deploy/remote/`)**: A powerful networking tool that allows a policy to run on a dedicated GPU server and be controlled remotely for evaluation.

## Architecture

The framework is designed around a clear separation of concerns, managed by the central configuration system.

```
+--------------------------+
|   Configuration (YAML)   |
| (Task, Policy, Training) |
+-------------+------------+
              |
              v
+-------------+------------+      +------------------------+
|      Training Script     |------>|   Policy (`nn.Module`)   |
|        (`train.py`)      |      +------------------------+
+-------------+------------+
              |
              v
+-------------+------------+      +------------------------+
|     Dataset Loaders      |<----->|   Raw Data (HDF5 etc.) |
| (`data_utils/datasets/`) |      +------------------------+
+--------------------------+

              +
              |
              v
+-------------+------------+
|    Evaluation Scripts    |
| (`vlastudio evaluate`, `vlastudio infer`) |
+-------------+------------+
              |
              v
+-------------+------------+      +------------------------+
|     MetaPolicy Wrapper     |------>|   Policy (`nn.Module`)   |
| (`benchmark/base.py`)    |      +------------------------+
+-------------+------------+
              |
+-------------v------------+      +------------------------+
|  Benchmark / Robot I/F   |------>| Sim Env / Real Robot   |
| (`benchmark/`, `deploy/`) |      +------------------------+
+--------------------------+
```

This modularity allows you to, for example, train a new policy on an existing dataset and then immediately evaluate it in both simulation and the real world by simply changing a few lines in a configuration file.
