"""Modular training and validation functions for YOLO keypoint detection models."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from ultralytics import YOLO
from .config import TrainingConfig, get_default_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOKeypointTrainer:
    """YOLO keypoint detection model trainer with modular training and validation capabilities."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer with configuration.
        
        Args:
            config: Training configuration. If None, uses default config.
        """
        self.config = config or get_default_config()
        self.model = None
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate training configuration."""
        if not Path(self.config.dataset_yaml_path).exists():
            logger.warning(f"Dataset YAML not found: {self.config.dataset_yaml_path}")
            logger.info("Please ensure dataset is properly configured before training")
        
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
    
    def load_model(self) -> YOLO:
        """Load YOLO keypoint detection model from configuration.
        
        Returns:
            Loaded YOLO model
        """
        logger.info(f"Loading keypoint detection model: {self.config.model_path}")
        self.model = YOLO(str(self.config.model_path))
        return self.model
    
    def train(self, **override_params) -> Dict[str, Any]:
        """Train the YOLO keypoint detection model.
        
        Args:
            **override_params: Parameters to override from config
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.load_model()
        
        # Prepare training parameters
        train_params = self.config.to_dict()
        train_params.update(override_params)
        
        logger.info(f"Starting keypoint detection training: {self.config.model_name}/{self.config.run}")
        logger.info(f"Dataset: {self.config.dataset_yaml_path}")
        logger.info(f"Epochs: {train_params['epochs']}, Batch size: {train_params['batch']}")
        logger.info(f"Keypoint shape: {train_params['kpt_shape']}, Pose loss weight: {train_params['pose']}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(train_params['project'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = self.model.train(**train_params)
            logger.info("Keypoint detection training completed successfully")
            return results
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate(self, model_path: Optional[str] = None, **override_params) -> Dict[str, Any]:
        """Validate the YOLO keypoint detection model.
        
        Args:
            model_path: Optional path to model for validation. If None, uses config model.
            **override_params: Parameters to override from config
            
        Returns:
            Validation results dictionary
        """
        # Load model for validation if different from training model
        if model_path:
            logger.info(f"Loading model for validation: {model_path}")
            val_model = YOLO(model_path)
        elif self.model is None:
            val_model = self.load_model()
        else:
            val_model = self.model
        
        # Prepare validation parameters
        val_params = {
            'data': str(self.config.dataset_yaml_path),
            'project': str(self.config.project_dir),
            'name': f'{self.config.run}_val'
        }
        val_params.update(override_params)
        
        logger.info(f"Starting keypoint detection validation: {val_params['name']}")
        logger.info(f"Dataset: {val_params['data']}")
        
        try:
            results = val_model.val(**val_params)
            logger.info("Keypoint detection validation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
    
    def train_and_validate(self, **override_params) -> Dict[str, Any]:
        """Train keypoint detection model and then validate it.
        
        Args:
            **override_params: Parameters to override from config
            
        Returns:
            Dictionary containing both training and validation results
        """
        logger.info("Starting keypoint detection training and validation pipeline")
        
        # Train the model
        train_results = self.train(**override_params)
        
        # Get the best model path from training results
        best_model_path = self.config.project_dir / self.config.run / "weights" / "best.pt"
        
        # Validate the trained model
        val_results = self.validate(model_path=str(best_model_path))
        
        logger.info("Keypoint detection training and validation pipeline completed")
        
        return {
            'training': train_results,
            'validation': val_results
        }
    
    def predict_keypoints(self, source: str, save: bool = True, **kwargs) -> Any:
        """Run keypoint detection inference on images or video.
        
        Args:
            source: Path to image, video, or directory
            save: Whether to save results
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Running keypoint detection inference on: {source}")
        
        try:
            results = self.model.predict(source=source, save=save, **kwargs)
            logger.info("Keypoint detection inference completed")
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current keypoint detection model and configuration.
        
        Returns:
            Dictionary with model and config information
        """
        if self.model is None:
            self.load_model()
        
        return {
            'model_path': str(self.config.model_path),
            'model_name': self.config.model_name,
            'run': self.config.run,
            'dataset': self.config.dataset_yaml_path,
            'epochs': self.config.epochs,
            'img_size': self.config.img_size,
            'batch_size': self.config.batch_size,
            'keypoint_shape': self.config.kpt_shape,
            'pose_loss_weight': self.config.pose_loss_weight,
            'output_dir': str(self.config.project_dir / self.config.run)
        }