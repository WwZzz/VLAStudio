#!/usr/bin/env python3
"""
Download OpenPI model checkpoints from Google Cloud Storage.

This script allows you to download pre-trained OpenPI models for fine-tuning or inference.
"""

import argparse
import os
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download


# Available models and their GCS paths
AVAILABLE_MODELS = {
    # Base models for fine-tuning
    "pi0_base": {
        "path": "gs://openpi-assets/checkpoints/pi0_base",
        "description": "Base π₀ model for fine-tuning"
    },
    "pi0_fast_base": {
        "path": "gs://openpi-assets/checkpoints/pi0_fast_base",
        "description": "Base π₀-FAST model (autoregressive) for fine-tuning"
    },
    "pi05_base": {
        "path": "gs://openpi-assets/checkpoints/pi05_base",
        "description": "Base π₀.₅ model for fine-tuning"
    },
    
    # Fine-tuned models for inference
    "pi0_fast_droid": {
        "path": "gs://openpi-assets/checkpoints/pi0_fast_droid",
        "description": "π₀-FAST fine-tuned on DROID dataset (inference ready)"
    },
    "pi0_droid": {
        "path": "gs://openpi-assets/checkpoints/pi0_droid",
        "description": "π₀ fine-tuned on DROID dataset"
    },
    "pi0_aloha_towel": {
        "path": "gs://openpi-assets/checkpoints/pi0_aloha_towel",
        "description": "π₀ fine-tuned for ALOHA towel folding task"
    },
    "pi0_aloha_tupperware": {
        "path": "gs://openpi-assets/checkpoints/pi0_aloha_tupperware",
        "description": "π₀ fine-tuned for ALOHA tupperware unpacking task"
    },
    "pi0_aloha_pen_uncap": {
        "path": "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap",
        "description": "π₀ fine-tuned for ALOHA pen uncapping task"
    },
    "pi05_libero": {
        "path": "gs://openpi-assets/checkpoints/pi05_libero",
        "description": "π₀.₅ fine-tuned for LIBERO benchmark"
    },
    "pi05_droid": {
        "path": "gs://openpi-assets/checkpoints/pi05_droid",
        "description": "π₀.₅ fine-tuned on DROID dataset with knowledge insulation"
    },
}


def list_available_models():
    """Print all available models with their descriptions."""
    print("\n=== Available OpenPI Models ===\n")
    print("Base Models (for fine-tuning):")
    for name, info in AVAILABLE_MODELS.items():
        if "base" in name:
            print(f"  • {name:20s} - {info['description']}")
    
    print("\nFine-tuned Models (for inference):")
    for name, info in AVAILABLE_MODELS.items():
        if "base" not in name:
            print(f"  • {name:20s} - {info['description']}")
    print()


def download_model(model_name: str, output_dir: str = None):
    """
    Download a specific model checkpoint.
    
    Args:
        model_name: Name of the model to download
        output_dir: Optional custom output directory
    """
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' not found.")
        print("\nAvailable models:")
        for name in AVAILABLE_MODELS.keys():
            print(f"  - {name}")
        return False
    
    model_info = AVAILABLE_MODELS[model_name]
    gcs_path = model_info["path"]
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Description: {model_info['description']}")
    print(f"Source: {gcs_path}")
    print(f"{'='*60}\n")
    
    try:
        # Set custom cache directory if specified
        original_cache_dir = os.environ.get("OPENPI_DATA_HOME")
        if output_dir is not None:
            os.environ["OPENPI_DATA_HOME"] = output_dir
            print(f"Using custom cache directory: {output_dir}")
        
        # Download the model
        checkpoint_dir = download.maybe_download(gcs_path)
        
        # Restore original cache directory setting
        if output_dir is not None:
            if original_cache_dir is not None:
                os.environ["OPENPI_DATA_HOME"] = original_cache_dir
            else:
                del os.environ["OPENPI_DATA_HOME"]
        
        print(f"\n✓ Successfully downloaded to: {checkpoint_dir}")
        print(f"\nYou can now use this model with:")
        print(f"  - For inference: Load from '{checkpoint_dir}'")
        print(f"  - For fine-tuning: Use as base model in your training config")
        return True
        
    except Exception as e:
        # Restore original cache directory setting on error
        if output_dir is not None:
            if original_cache_dir is not None:
                os.environ["OPENPI_DATA_HOME"] = original_cache_dir
            else:
                if "OPENPI_DATA_HOME" in os.environ:
                    del os.environ["OPENPI_DATA_HOME"]
        
        print(f"\n✗ Error downloading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenPI model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python download.py --list
  
  # Download a specific model
  python download.py --model pi05_droid
  
  # Download to a custom directory
  python download.py --model pi0_base --output /path/to/models
  
  # Download multiple models
  python download.py --model pi0_base pi05_droid
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        nargs='+',
        help="Name(s) of the model(s) to download"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available models"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Custom output directory (default: ~/.cache/openpi)"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        list_available_models()
        return
    
    # Check if model name is provided
    if not args.model:
        print("Error: Please specify a model to download with --model or use --list to see available models")
        parser.print_help()
        return
    
    # Download requested model(s)
    success_count = 0
    for model_name in args.model:
        if download_model(model_name, args.output):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Downloaded {success_count}/{len(args.model)} model(s) successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
