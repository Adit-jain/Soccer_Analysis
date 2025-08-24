"""Configuration file for training YOLO keypoint detection models."""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))


@dataclass
class TrainingConfig:
    """Configuration class for YOLO keypoint detection model training."""
    
    # Dataset configuration
    dataset_yaml_path: str = r"F:\Datasets\SoccerNet\Data\calibration\dataset.yaml"
    
    # Model configuration
    model_name: str = 'yolov11_keypoints'
    run: str = 'First'
    start_model_name: str = 'yolov11_keypoints'
    start_model_run: str = 'First'
    
    # Training hyperparameters
    epochs: int = 150
    img_size: int = 640
    batch_size: int = 32
    lr0: float = 0.01
    lrf: float = 0.01
    dropout: float = 0.3
    
    # Training settings
    resume: bool = True
    save_period: int = 1
    single_cls: bool = True  # Single class for pitch/field
    freeze: bool = False
    seed: int = 44
    plots: bool = True

    # Augmentation settings
    fliplr: float = 0.0
    mosaic: float = 0.3
    
    # Hardware settings
    device: int = 0
    
    def __post_init__(self):
        """Set model path based on start_model_name."""
        if 'original_' in self.start_model_name:
            model_file = f"{self.start_model_name.split('original_')[-1]}.pt"
            self.model_path = PROJECT_DIR / f"Models/Pretrained/{model_file}"
        else:
            self.model_path = PROJECT_DIR / f"Models/Trained/{self.start_model_name}/{self.start_model_run}/weights/best.pt"
    
    @property
    def project_dir(self) -> Path:
        """Get the project directory for saving models."""
        return PROJECT_DIR / f'Models/Trained/{self.model_name}'
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for YOLO training."""
        return {
            'data': str(self.dataset_yaml_path),
            'epochs': self.epochs,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'project': str(self.project_dir),
            'name': self.run,
            'seed': self.seed,
            'resume': self.resume,
            'save': True,
            'save_period': self.save_period,
            'single_cls': self.single_cls,
            'freeze': self.freeze,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'dropout': self.dropout,
            'fliplr': self.fliplr,
            'mosaic': self.mosaic,
            'plots': self.plots,
            'device': self.device,
        }


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def create_custom_config(**kwargs) -> TrainingConfig:
    """Create custom training configuration with overrides."""
    config = TrainingConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    # Trigger post_init to update model_path if start_model_name changed
    config.__post_init__()
    return config