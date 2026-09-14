"""
Unified logging system for ILStudio using loguru.

Loguru provides a global logger that can be imported anywhere without initialization:

    from loguru import logger
    
    logger.info("Training started")
    logger.warning("Low memory")
    logger.error("Failed to load checkpoint")
    logger.debug("Detailed debug info")

This module configures the global loguru logger and intercepts standard logging
to ensure compatibility with HuggingFace Transformers and other libraries.
"""

import os
# Suppress TensorFlow warnings and info messages BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations messages
from loguru import logger
import sys
import logging
import warnings

# Remove default handler
logger.remove()

# Add custom handler with simple format
logger.add(
    sys.stdout,
    format="<level>[{level}]</level> {message}",
    level="INFO",
    colorize=True,
)

# Intercept standard logging to redirect to loguru
# This ensures compatibility with HuggingFace Transformers and other libraries using standard logging
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Intercept standard logging
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Reduce verbosity of specific libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# Suppress tokenizers parallelism warnings
logging.getLogger("huggingface.tokenizers").setLevel(logging.ERROR)

# Suppress TensorFlow logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("tensorboard").setLevel(logging.WARNING)

# Suppress other noisy libraries
logging.getLogger("absl").setLevel(logging.ERROR)  # TensorFlow uses absl for logging

# Suppress tokenizers parallelism warnings (these are emitted via warnings module, not logging)
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The current process just got forked.*", category=UserWarning)

# Export for convenience
__all__ = ['logger']
