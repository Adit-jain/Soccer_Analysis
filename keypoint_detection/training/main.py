#!/usr/bin/env python3
"""Main entry point for YOLO keypoint detection model training and validation.

This script provides a simple command-line interface for training and validating
YOLO keypoint detection models with customizable parameters.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from .config import create_custom_config
from .trainer import YOLOKeypointTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and validate YOLO keypoint detection models for soccer field analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main action
    parser.add_argument(
        'action',
        choices=['train', 'validate', 'both', 'predict'],
        help='Action to perform'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        default='yolov11_keypoints',
        help='Model name for saving'
    )
    
    parser.add_argument(
        '--run',
        default='First',
        help='Run identifier'
    )
    
    parser.add_argument(
        '--start-model',
        default='original_yolo11n-pose',
        help='Starting model name (use pose models for keypoint detection)'
    )
    
    parser.add_argument(
        '--model-path',
        help='Path to specific model file (overrides start-model)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )
    
    parser.add_argument(
        '--pose-loss-weight',
        type=float,
        default=5.0,
        help='Weight for pose/keypoint loss'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--data',
        default=r"F:\Datasets\SoccerNet\Data\calibration\keypoints_dataset.yaml",
        help='Path to keypoint dataset YAML file'
    )
    
    # Training options
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    parser.add_argument(
        '--freeze',
        action='store_true',
        help='Freeze backbone layers'
    )
    
    parser.add_argument(
        '--single-cls',
        action='store_true',
        default=True,
        help='Train as single class (field/pitch)'
    )
    
    # Validation specific
    parser.add_argument(
        '--val-model',
        help='Specific model path for validation'
    )
    
    # Prediction specific
    parser.add_argument(
        '--source',
        help='Source for prediction (image, video, or directory path)'
    )
    
    parser.add_argument(
        '--save-pred',
        action='store_true',
        default=True,
        help='Save prediction results'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Create configuration
        config_params = {
            'model_name': args.model_name,
            'run': args.run,
            'start_model_name': args.start_model,
            'epochs': args.epochs,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'lr0': args.lr0,
            'dropout': args.dropout,
            'pose_loss_weight': args.pose_loss_weight,
            'dataset_yaml_path': args.data,
            'resume': args.resume,
            'freeze': args.freeze,
            'single_cls': args.single_cls
        }
        
        config = create_custom_config(**config_params)
        
        # Override model path if provided
        if args.model_path:
            config.model_path = Path(args.model_path)
        
        # Create trainer
        trainer = YOLOKeypointTrainer(config)
        
        print(f"🎯 Starting keypoint detection {args.action} with configuration:")
        info = trainer.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
        # Execute requested action
        if args.action == 'train':
            print("🚀 Starting keypoint detection training...")
            results = trainer.train()
            print("✅ Keypoint detection training completed successfully!")
            
        elif args.action == 'validate':
            print("🔍 Starting keypoint detection validation...")
            model_path = args.val_model or str(config.model_path)
            results = trainer.validate(model_path=model_path)
            print("✅ Keypoint detection validation completed successfully!")
            
        elif args.action == 'both':
            print("🚀 Starting keypoint detection training and validation pipeline...")
            results = trainer.train_and_validate()
            print("✅ Keypoint detection training and validation completed successfully!")
            
        elif args.action == 'predict':
            if not args.source:
                print("❌ Error: --source is required for prediction")
                sys.exit(1)
            print(f"🎯 Running keypoint detection inference...")
            results = trainer.predict_keypoints(
                source=args.source,
                save=args.save_pred
            )
            print("✅ Keypoint detection inference completed successfully!")
        
        print("\n🎉 All keypoint detection operations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()