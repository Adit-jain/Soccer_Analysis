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
        help='Model name for saving'
    )
    
    parser.add_argument(
        '--run',
        help='Run identifier'
    )
    
    parser.add_argument(
        '--start-model',
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
    
    # Dataset configuration
    parser.add_argument(
        '--data',
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