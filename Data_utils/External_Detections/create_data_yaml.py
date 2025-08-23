"""Create YOLO dataset configuration file.

Simple script to generate data.yaml file for Ultralytics YOLO training.
"""

import os
from pathlib import Path
import yaml

def create_dataset_yaml(dataset_root: str = r'D:\Datasets\SoccerAnalysis_Final\V1') -> None:
    """Create YOLO dataset configuration file.
    
    Args:
        dataset_root: Root directory of the dataset
    """
    data = {
        'path': dataset_root,
        'train': 'images/train',
        'val': 'images/test',
        'names': {
            0: 'Player',
            1: 'Ball',
            2: 'Referee'
        }
    }

    yaml_path = Path(dataset_root) / 'data.yaml'
    
    with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    
    print(f"✓ Dataset YAML created at: {yaml_path}")


if __name__ == '__main__':
    create_dataset_yaml()