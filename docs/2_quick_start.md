# 2. Quick Start

This guide will walk you through the basic steps to get IL-Studio up and running, from installation to running a pre-trained model in simulation.

## 1. Installation

### Requirements
A full list of dependencies can be found in `requirements.txt`. Key requirements include:
*   Python 3.8+
*   PyTorch
*   TensorFlow (for data loading)
*   OpenCV

### Setup Steps
We recommend setting up a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-org/IL-Studio.git
cd IL-Studio

# Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Download Pre-trained Models & Datasets

We provide example scripts to download pre-trained models and the necessary datasets for evaluation.

*(Note: The actual download script is not present in the codebase, so a placeholder is provided. You would typically provide a script to fetch these from a cloud storage bucket.)*
```bash
# Download pre-trained models
# ./scripts/download_models.sh

# Download demonstration data
# ./scripts/download_data.sh
```

For this guide, we will assume a pre-trained ACT model exists at `ckpt/act_sim_transfer_cube_scripted_zscore_example`.

## 3. Run Your First Evaluation

Once the models are available, you can run an evaluation in the `aloha` simulation environment. This task involves a bimanual robot transferring a cube.

```bash
# Run the simulation evaluation script for 4 episodes
vlastudio evaluate --runtime current \
    -m ckpt/act_sim_transfer_cube_scripted_zscore_example \
    -e aloha \
    --num_rollout 4 \
    --output_dir results/quick_start_eval
```

### What to Expect
*   The script will load the policy and the `aloha` environment.
*   A MuJoCo simulation window should appear.
*   The robot arms will attempt to pick up a cube with one arm and pass it to the other.
*   After the evaluation is complete, a summary of the success rate will be printed to the console.
*   A video of the rollouts will be saved to `results/quick_start_eval/aloha/video/`.

Congratulations! You have successfully run your first evaluation in IL-Studio.
