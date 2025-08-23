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

from keypoint_detection.training.config import create_custom_config
from keypoint_detection.training.trainer import YOLOKeypointTrainer
import traceback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and validate YOLO keypoint detection models for soccer field analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main action
    parser.add_argument(
        '--action',
        choices=['train', 'validate', 'both', 'predict'],
        help='Action to perform',
        default = 'train'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        help='Model name for saving'
    )
    
    parser.add_argument(
        '--run',
        help='Run identifier'
    )
    
    parser.add_argument(
        '--start-model',
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
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        help='Input image size'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr0',
        type=float,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate'
    )
    
    parser.add_argument(
        '--pose-loss-weight',
        type=float,
        help='Weight for pose/keypoint loss'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--data',
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
        help='Save prediction results'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Create configuration with only provided arguments
        config_params = {}
        if args.model_name is not None:
            config_params['model_name'] = args.model_name
        if args.run is not None:
            config_params['run'] = args.run
        if args.start_model is not None:
            config_params['start_model_name'] = args.start_model
        if args.epochs is not None:
            config_params['epochs'] = args.epochs
        if args.img_size is not None:
            config_params['img_size'] = args.img_size
        if args.batch_size is not None:
            config_params['batch_size'] = args.batch_size
        if args.lr0 is not None:
            config_params['lr0'] = args.lr0
        if args.dropout is not None:
            config_params['dropout'] = args.dropout
        if args.pose_loss_weight is not None:
            config_params['pose_loss_weight'] = args.pose_loss_weight
        if args.data is not None:
            config_params['dataset_yaml_path'] = args.data
        if args.resume:
            config_params['resume'] = args.resume
        if args.freeze:
            config_params['freeze'] = args.freeze
        if args.single_cls:
            config_params['single_cls'] = args.single_cls
        
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
                save=args.save_pred if args.save_pred else True
            )
            print("✅ Keypoint detection inference completed successfully!")
        
        print("\n🎉 All keypoint detection operations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()