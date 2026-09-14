#!/bin/bash
set -e

# ============================================================
# BEHAVIOR-1K Installation Script for ILStudio (using uv)
# ============================================================

# Parse arguments
HELP=false
NEW_ENV=false
OMNIGIBSON=false
BDDL=false
JOYLO=false
DATASET=false
PRIMITIVES=false
EVAL=false
DEV=false
CUDA_VERSION="12.4"
ACCEPT_NVIDIA_EULA=false
ACCEPT_DATASET_TOS=false

[ "$#" -eq 0 ] && HELP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) HELP=true; shift ;;
        --new-env) NEW_ENV=true; shift ;;
        --omnigibson) OMNIGIBSON=true; shift ;;
        --bddl) BDDL=true; shift ;;
        --joylo) JOYLO=true; shift ;;
        --dataset) DATASET=true; shift ;;
        --primitives) PRIMITIVES=true; shift ;;
        --eval) EVAL=true; shift ;;
        --dev) DEV=true; shift ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --accept-nvidia-eula) ACCEPT_NVIDIA_EULA=true; shift ;;
        --accept-dataset-tos) ACCEPT_DATASET_TOS=true; shift ;;
        --all) NEW_ENV=true; OMNIGIBSON=true; BDDL=true; JOYLO=true; DATASET=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$HELP" = true ]; then
    cat << EOF
BEHAVIOR-1K Installation Script for ILStudio (using uv)
Usage: ./setup.sh [OPTIONS]

Options:
  -h, --help              Display this help message
  --new-env               Create a new uv virtual environment
  --omnigibson            Install OmniGibson + Isaac Sim (core physics simulator)
  --bddl                  Install BDDL (Behavior Domain Definition Language)
  --joylo                 Install JoyLo (teleoperation interface)
  --dataset               Download BEHAVIOR datasets (requires --omnigibson)
  --primitives            Install OmniGibson with primitives support
  --eval                  Install evaluation dependencies
  --dev                   Install development dependencies
  --cuda-version VERSION  Specify CUDA version (default: 12.4)
  --accept-nvidia-eula    Automatically accept NVIDIA Isaac Sim EULA
  --accept-dataset-tos    Automatically accept BEHAVIOR Dataset Terms
  --all                   Install everything (--new-env --omnigibson --bddl --joylo --dataset)

Example: ./setup.sh --new-env --omnigibson --bddl --joylo --dataset
Example: ./setup.sh --all --accept-nvidia-eula --accept-dataset-tos
EOF
    exit 0
fi

# Validate dependencies
[ "$OMNIGIBSON" = true ] && [ "$BDDL" = false ] && { echo "ERROR: --omnigibson requires --bddl"; exit 1; }
[ "$PRIMITIVES" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --primitives requires --omnigibson"; exit 1; }
[ "$EVAL" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --eval requires --omnigibson"; exit 1; }
[ "$DATASET" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --dataset requires --omnigibson"; exit 1; }

# Get script directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BEHAVIOR_1K_DIR="$SCRIPT_DIR/BEHAVIOR-1K"

# Check BEHAVIOR-1K directory exists
[ ! -d "$BEHAVIOR_1K_DIR" ] && { echo "ERROR: BEHAVIOR-1K directory not found at $BEHAVIOR_1K_DIR"; exit 1; }

# Check and install system dependencies for Isaac Sim
check_and_install_system_deps() {
    echo "Checking system dependencies for Isaac Sim..."
    
    MISSING_LIBS=""
    
    # Check for libGLU (most common missing library)
    if ! ldconfig -p | grep -q "libGLU.so.1"; then
        MISSING_LIBS="$MISSING_LIBS libglu1-mesa"
    fi
    
    # Check for other OpenGL libraries
    if ! ldconfig -p | grep -q "libGL.so.1"; then
        MISSING_LIBS="$MISSING_LIBS libgl1-mesa-glx"
    fi
    
    if ! ldconfig -p | grep -q "libEGL.so.1"; then
        MISSING_LIBS="$MISSING_LIBS libegl1-mesa"
    fi
    
    if [ -n "$MISSING_LIBS" ]; then
        echo "Missing system libraries detected:$MISSING_LIBS"
        echo "Attempting to install..."
        
        if command -v apt-get &> /dev/null; then
            if [ "$EUID" -eq 0 ]; then
                apt-get update && apt-get install -y $MISSING_LIBS libxrender1 libsm6
            else
                echo "WARNING: Cannot install system libraries without root. Please run:"
                echo "  sudo apt-get install -y$MISSING_LIBS libxrender1 libsm6"
            fi
        else
            echo "WARNING: apt-get not found. Please install the following libraries manually:"
            echo "  $MISSING_LIBS libxrender1 libsm6"
        fi
    else
        echo "✓ All required system libraries are present"
    fi
}

# Check uv is installed
command -v uv >/dev/null || { echo "ERROR: uv not found. Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# Function to prompt for NVIDIA EULA
prompt_nvidia_eula() {
    if [ "$ACCEPT_NVIDIA_EULA" = true ]; then
        return 0
    fi
    
    echo ""
    echo "=== NVIDIA ISAAC SIM EULA ==="
    echo "Installing OmniGibson requires acceptance of the NVIDIA Isaac Sim EULA."
    echo "See: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement"
    echo ""
    echo "Do you accept the NVIDIA Isaac Sim EULA? (y/N)"
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "NVIDIA EULA not accepted. Installation cancelled."
        exit 1
    fi
    ACCEPT_NVIDIA_EULA=true
}

# Function to prompt for dataset TOS
prompt_dataset_tos() {
    if [ "$ACCEPT_DATASET_TOS" = true ]; then
        return 0
    fi
    
    echo ""
    echo "=== BEHAVIOR DATA BUNDLE END USER LICENSE AGREEMENT ==="
    cat << EOF
Last revision: December 8, 2022

This License Agreement is for the BEHAVIOR Data Bundle ("Data"). 
The Data may only be used for non-commercial academic research.
You are strictly prohibited from extracting any Data or reverse engineering.
You may not redistribute the key or any other Data or elements in whole or part.

THE DATA AND SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
EOF
    echo ""
    echo "Do you accept the BEHAVIOR Dataset Terms? (y/N)"
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Dataset terms not accepted. Installation cancelled."
        exit 1
    fi
    ACCEPT_DATASET_TOS=true
}

# Helper function to check GLIBC version
check_glibc_old() {
    ldd --version 2>&1 | grep -qE "2\.(31|32|33)"
}

# Function to install Isaac Sim packages
install_isaac_sim() {
    if python -c "import isaacsim" 2>/dev/null; then
        echo "✓ Isaac Sim already installed, skipping..."
        return 0
    fi
    
    echo "Installing Isaac Sim via pip..."
    
    local temp_dir=$(mktemp -d)
    local packages=(
        "omniverse_kit-106.5.0.162521"
        "isaacsim_kernel-4.5.0.0"
        "isaacsim_app-4.5.0.0"
        "isaacsim_core-4.5.0.0"
        "isaacsim_gui-4.5.0.0"
        "isaacsim_utils-4.5.0.0"
        "isaacsim_storage-4.5.0.0"
        "isaacsim_asset-4.5.0.0"
        "isaacsim_sensor-4.5.0.0"
        "isaacsim_robot_motion-4.5.0.0"
        "isaacsim_robot-4.5.0.0"
        "isaacsim_benchmark-4.5.0.0"
        "isaacsim_code_editor-4.5.0.0"
        "isaacsim_ros1-4.5.0.0"
        "isaacsim_cortex-4.5.0.0"
        "isaacsim_example-4.5.0.0"
        "isaacsim_replicator-4.5.0.0"
        "isaacsim_rl-4.5.0.0"
        "isaacsim_robot_setup-4.5.0.0"
        "isaacsim_ros2-4.5.0.0"
        "isaacsim_template-4.5.0.0"
        "isaacsim_test-4.5.0.0"
        "isaacsim-4.5.0.0"
        "isaacsim_extscache_physics-4.5.0.0"
        "isaacsim_extscache_kit-4.5.0.0"
        "isaacsim_extscache_kit_sdk-4.5.0.0"
    )
    
    local wheel_files=()
    for pkg in "${packages[@]}"; do
        local pkg_name=${pkg%-*}
        local filename="${pkg}-cp310-none-manylinux_2_34_x86_64.whl"
        local url="https://pypi.nvidia.com/${pkg_name//_/-}/$filename"
        local filepath="$temp_dir/$filename"
        
        echo "Downloading $pkg..."
        if ! curl -sL "$url" -o "$filepath"; then
            echo "ERROR: Failed to download $pkg"
            rm -rf "$temp_dir"
            return 1
        fi
        
        # Rename for older GLIBC
        if check_glibc_old; then
            local new_filepath="${filepath/manylinux_2_34/manylinux_2_31}"
            mv "$filepath" "$new_filepath"
            filepath="$new_filepath"
        fi
        
        wheel_files+=("$filepath")
    done
    
    echo "Installing Isaac Sim packages..."
    uv pip install "${wheel_files[@]}"
    rm -rf "$temp_dir"
    
    # Verify installation
    if ! python -c "import isaacsim" 2>/dev/null; then
        echo "ERROR: Isaac Sim installation verification failed"
        return 1
    fi
    
    # Fix websockets conflict
    local ISAAC_PATH=$(python -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))" 2>/dev/null)
    if [ -n "$ISAAC_PATH" ] && [ -d "$ISAAC_PATH/extscache" ]; then
        echo "Fixing websockets conflict..."
        find "$ISAAC_PATH/extscache" -type d -name "websockets" -path "*/pip_prebundle/*" -exec rm -rf {} + 2>/dev/null || true
    fi
    
    echo "✓ Isaac Sim installation completed"
}

# ============================================================
# Main Installation
# ============================================================

cd "$SCRIPT_DIR"

# Create virtual environment
if [ "$NEW_ENV" = true ]; then
    echo "Creating uv virtual environment..."
    
    # Remove existing .venv if exists
    if [ -d ".venv" ]; then
        echo "Removing existing .venv..."
        rm -rf .venv
    fi
    
    uv venv --python 3.10
    source .venv/bin/activate
    
    # Verify Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    [ "$PYTHON_VERSION" != "3.10" ] && { echo "ERROR: Python 3.10 required, found $PYTHON_VERSION"; exit 1; }
    
    # Install basic dependencies
    echo "Installing numpy and setuptools..."
    uv pip install "numpy<2" "setuptools<=79"
    
    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA $CUDA_VERSION support..."
    CUDA_VER_SHORT=$(echo $CUDA_VERSION | sed 's/\.//g')
    uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}
    
    echo "✓ Virtual environment created and basic dependencies installed"
else
    # Activate existing environment if exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        echo "WARNING: No .venv found. Using current Python environment."
    fi
fi

# Install BDDL
if [ "$BDDL" = true ]; then
    echo "Installing BDDL..."
    [ ! -d "$BEHAVIOR_1K_DIR/bddl3" ] && { echo "ERROR: bddl3 directory not found"; exit 1; }
    uv pip install -e "$BEHAVIOR_1K_DIR/bddl3"
    echo "✓ BDDL installed"
fi

# Install OmniGibson with Isaac Sim
if [ "$OMNIGIBSON" = true ]; then
    echo "Installing OmniGibson..."
    [ ! -d "$BEHAVIOR_1K_DIR/OmniGibson" ] && { echo "ERROR: OmniGibson directory not found"; exit 1; }
    
    # Check and install system dependencies
    check_and_install_system_deps
    
    # Prompt for NVIDIA EULA
    prompt_nvidia_eula
    export OMNI_KIT_ACCEPT_EULA=YES
    
    # Check for conflicting environment variables
    if [[ -n "$EXP_PATH" || -n "$CARB_APP_PATH" || -n "$ISAAC_PATH" ]]; then
        echo "WARNING: Found existing Isaac Sim environment variables."
        echo "You may need to unset EXP_PATH, CARB_APP_PATH, and ISAAC_PATH if issues occur."
    fi
    
    # Install OmniGibson
    uv pip install -e "$BEHAVIOR_1K_DIR/OmniGibson"
    
    # Install Isaac Sim
    install_isaac_sim || { echo "ERROR: Isaac Sim installation failed"; exit 1; }
    
    # Force reinstall cffi to resolve compatibility issues
    uv pip install --force-reinstall cffi==1.17.1
    
    echo "✓ OmniGibson + Isaac Sim installed"
fi

# Install JoyLo
if [ "$JOYLO" = true ]; then
    echo "Installing JoyLo..."
    [ ! -d "$BEHAVIOR_1K_DIR/joylo" ] && { echo "ERROR: joylo directory not found"; exit 1; }
    uv pip install -e "$BEHAVIOR_1K_DIR/joylo"
    echo "✓ JoyLo installed"
fi

# Install primitives support
if [ "$PRIMITIVES" = true ]; then
    echo "Installing primitives support..."
    uv pip install -e "$BEHAVIOR_1K_DIR/OmniGibson[primitives]"
    echo "✓ Primitives support installed"
fi

# Install eval support
if [ "$EVAL" = true ]; then
    echo "Installing evaluation support..."
    uv pip install -e "$BEHAVIOR_1K_DIR/OmniGibson[eval]"
    
    # Install torch-cluster
    TORCH_VERSION=$(pip show torch | grep Version | cut -d " " -f 2)
    uv pip install torch-cluster -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}.html"
    
    # Install av
    uv pip install av
    echo "✓ Evaluation support installed"
fi

# Install dev support
if [ "$DEV" = true ]; then
    echo "Installing development dependencies..."
    uv pip install -e "$BEHAVIOR_1K_DIR/OmniGibson[dev]"
    
    # Setup pre-commit
    cd "$BEHAVIOR_1K_DIR/OmniGibson"
    uv pip install pre-commit
    pre-commit install
    cd "$SCRIPT_DIR"
    
    echo "✓ Development dependencies installed"
fi

# Install datasets
if [ "$DATASET" = true ]; then
    python -c "import omnigibson" || {
        echo "ERROR: OmniGibson import failed. Please make sure OmniGibson is installed correctly."
        exit 1
    }
    
    # Prompt for dataset TOS
    prompt_dataset_tos
    
    echo "Downloading datasets..."
    export OMNI_KIT_ACCEPT_EULA=YES
    
    DATASET_ACCEPT_FLAG="True"
    if [ "$ACCEPT_DATASET_TOS" = false ]; then
        DATASET_ACCEPT_FLAG="False"
    fi
    
    echo "Downloading OmniGibson robot assets..."
    python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()" || {
        echo "ERROR: OmniGibson robot assets download failed"
        exit 1
    }
    
    echo "Downloading BEHAVIOR-1K assets..."
    python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=${DATASET_ACCEPT_FLAG})" || {
        echo "ERROR: BEHAVIOR-1K assets download failed"
        exit 1
    }
    
    echo "Downloading 2025 BEHAVIOR Challenge Task Instances..."
    python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()" || {
        echo "ERROR: 2025 BEHAVIOR Challenge Task Instances download failed"
        exit 1
    }
    
    echo "✓ Datasets downloaded"
fi

# Install behavior1k package itself
if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    echo "Installing behavior1k package..."
    uv pip install -e "$SCRIPT_DIR"
    echo "✓ behavior1k package installed"
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "=== Installation Complete! ==="
if [ "$NEW_ENV" = true ]; then echo "✓ Created uv virtual environment (.venv)"; fi
if [ "$BDDL" = true ]; then echo "✓ Installed BDDL"; fi
if [ "$OMNIGIBSON" = true ]; then echo "✓ Installed OmniGibson + Isaac Sim"; fi
if [ "$JOYLO" = true ]; then echo "✓ Installed JoyLo"; fi
if [ "$PRIMITIVES" = true ]; then echo "✓ Installed primitives support"; fi
if [ "$EVAL" = true ]; then echo "✓ Installed evaluation support"; fi
if [ "$DEV" = true ]; then echo "✓ Installed development dependencies"; fi
if [ "$DATASET" = true ]; then echo "✓ Downloaded datasets"; fi
echo ""
echo "To activate the environment: source .venv/bin/activate"
echo ""

