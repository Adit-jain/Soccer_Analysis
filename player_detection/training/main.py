#!/usr/bin/env python3
"""Main entry point for YOLO model training and validation.

This script provides a simple command-line interface for training and validating
YOLO models with customizable parameters.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from .config import create_custom_config
from .trainer import YOLOTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and validate YOLO models for soccer analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main action
    parser.add_argument(
        'action',
        choices=['train', 'validate', 'both'],
        help='Action to perform'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        default='yolov11_sahi_1280',
        help='Model name for saving'
    )
    
    parser.add_argument(
        '--run',
        default='First',
        help='Run identifier'
    )
    
    parser.add_argument(
        '--start-model',
        default='original_yolo11n',
        help='Starting model name'
    )
    
    parser.add_argument(
        '--model-path',
        help='Path to specific model file (overrides start-model)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=1280,
        help='Input image size'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
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
        default=0.3,
        help='Dropout rate'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--data',
        default=r"F:\Datasets\SoccerAnalysis_Final\V1\data.yaml",
        help='Path to dataset YAML file'
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
        help='Train as single class'
    )
    
    # Validation specific
    parser.add_argument(
        '--val-model',
        help='Specific model path for validation'
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
        trainer = YOLOTrainer(config)
        
        print(f"🏃 Starting {args.action} with configuration:")
        info = trainer.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
        # Execute requested action
        if args.action == 'train':
            print("🚀 Starting training...")
            results = trainer.train()
            print("✅ Training completed successfully!")
            
        elif args.action == 'validate':
            print("🔍 Starting validation...")
            model_path = args.val_model or str(config.model_path)
            results = trainer.validate(model_path=model_path)
            print("✅ Validation completed successfully!")
            
        elif args.action == 'both':
            print("🚀 Starting training and validation pipeline...")
            results = trainer.train_and_validate()
            print("✅ Training and validation completed successfully!")
        
        print("\n🎉 All operations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()