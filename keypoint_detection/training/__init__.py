"""Training module for keypoint detection models."""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from keypoint_detection.training.config import get_default_config, create_custom_config, TrainingConfig
from keypoint_detection.training.trainer import YOLOKeypointTrainer

__all__ = ['get_default_config', 'create_custom_config', 'TrainingConfig', 'YOLOKeypointTrainer']