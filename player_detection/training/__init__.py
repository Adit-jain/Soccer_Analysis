"""Training package for YOLO model training and validation.

This package provides a modern, modular approach to training YOLO models
for soccer analysis with configurable parameters and reusable components.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from player_detection.training.config import TrainingConfig, get_default_config, create_custom_config
from player_detection.training.trainer import YOLOTrainer, quick_train, quick_validate

__all__ = [
    'TrainingConfig',
    'get_default_config', 
    'create_custom_config',
    'YOLOTrainer',
    'quick_train',
    'quick_validate'
]

__version__ = '1.0.0'