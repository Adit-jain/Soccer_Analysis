"""Training package for YOLO model training and validation.

This package provides a modern, modular approach to training YOLO models
for soccer analysis with configurable parameters and reusable components.
"""

from .config import TrainingConfig, get_default_config, create_custom_config
from .trainer import YOLOTrainer, quick_train, quick_validate

__all__ = [
    'TrainingConfig',
    'get_default_config', 
    'create_custom_config',
    'YOLOTrainer',
    'quick_train',
    'quick_validate'
]

__version__ = '1.0.0'