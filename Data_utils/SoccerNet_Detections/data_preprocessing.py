"""SoccerNet dataset preprocessing for YOLO format.

Processes SoccerNet tracking dataset ground truth files and converts them
to YOLO format annotations with proper train/val/test splits.
"""

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

from constants import dataset_dir

# Suppress pandas chained assignment warnings
pd.options.mode.chained_assignment = None

# Directory structure
BASE_DIR = Path(dataset_dir) / 'tracking'
IMAGES_PATH = BASE_DIR / 'images'
LABELS_PATH = BASE_DIR / 'labels'

# Dataset split tracking
train_image_folders: List[str] = []
val_image_folders: List[str] = []
test_image_folders: List[str] = []


def read_ground_truth_file(ground_truth_path: str) -> pd.DataFrame:
    """Process SoccerNet ground truth file and convert to YOLO format.
    
    Args:
        ground_truth_path: Path to the ground truth .txt file
        
    Returns:
        DataFrame with YOLO format annotations (normalized coordinates)
    """
    try:
        # Read ground truth data (MOT format)
        raw_gt_data = pd.read_csv(ground_truth_path, header=None)
        raw_gt_data.columns = [
            'frame', 'object_id', 'x', 'y', 'w', 'h', 
            'conf', 'class_id', 'visibility', 'unknown'
        ]
        
        # Initialize all objects as players (class 0)
        raw_gt_data['class'] = 0
        raw_gt_data['area'] = raw_gt_data['w'] * raw_gt_data['h']
        
        # Select relevant columns
        ground_truth_data = raw_gt_data[
            ['class', 'x', 'y', 'w', 'h', 'frame', 'area']
        ].copy()

        # Convert to YOLO format (normalized coordinates)
        # Assuming image dimensions of 1920x1080
        IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080
        
        # Convert from top-left corner to center coordinates and normalize
        ground_truth_data['x'] = (ground_truth_data['x'] + ground_truth_data['w'] / 2) / IMAGE_WIDTH
        ground_truth_data['y'] = (ground_truth_data['y'] + ground_truth_data['h'] / 2) / IMAGE_HEIGHT
        ground_truth_data['w'] = ground_truth_data['w'] / IMAGE_WIDTH
        ground_truth_data['h'] = ground_truth_data['h'] / IMAGE_HEIGHT
        
        return ground_truth_data
        
    except Exception as e:
        print(f"Error reading ground truth file {ground_truth_path}: {e}")
        return pd.DataFrame()


def process_soccernet_dataset() -> None:
    """Process the complete SoccerNet dataset and generate YOLO format annotations."""
    print("📎 Processing SoccerNet dataset...")
    
    if not IMAGES_PATH.exists():
        print(f"Error: Images directory not found: {IMAGES_PATH}")
        return
    
    # Process each dataset split
    for data_type in os.listdir(IMAGES_PATH):
        data_type_path = IMAGES_PATH / data_type
        
        if not data_type_path.is_dir():
            continue
            
        print(f"\n📁 Processing {data_type} split: {data_type_path}")

        # Process each video sequence
        video_dirs = [d for d in os.listdir(data_type_path) if (data_type_path / d).is_dir()]
        for vid_name in tqdm(video_dirs, desc=f"Processing {data_type} videos"):
            vid_path = data_type_path / vid_name
            image_folder_path = vid_path / 'img1'
            
            if not image_folder_path.exists():
                print(f"Warning: Image folder not found: {image_folder_path}")
                continue

            # Track folder paths for dataset YAML
            relative_path = str(image_folder_path.relative_to(BASE_DIR)).replace('\\', '/')
            if data_type == 'train':
                train_image_folders.append(relative_path)
            elif data_type == 'test':
                val_image_folders.append(relative_path)
            elif data_type == 'challenge':
                test_image_folders.append(relative_path)

            # Process ground truth data (only for train/test, not challenge)
            if data_type in ['train', 'test']:
                gt_path = vid_path / 'gt' / 'gt.txt'
                
                if not gt_path.exists():
                    print(f"Warning: Ground truth file not found: {gt_path}")
                    continue
                    
                gt_data = read_ground_truth_file(str(gt_path))
                
                if gt_data.empty:
                    continue

                # Process each image in the sequence
                image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')]
                for image_file in image_files:
                    try:
                        image_number = int(Path(image_file).stem)
                        
                        # Filter annotations for this specific frame
                        frame_annotations = gt_data[gt_data['frame'] == image_number].copy()
                        
                        if len(frame_annotations) > 0:
                            # Identify the ball as the smallest object (by area)
                            min_area_idx = frame_annotations['area'].idxmin()
                            frame_annotations.loc[min_area_idx, 'class'] = 1  # Ball class
                            
                            # Remove helper columns
                            frame_annotations = frame_annotations.drop(['frame', 'area'], axis=1)

                            # Create label file path
                            label_path = str(image_folder_path).replace('images', 'labels')
                            label_path = Path(label_path) / f"{Path(image_file).stem}.txt"
                            
                            # Ensure label directory exists
                            label_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Save annotations
                            frame_annotations.to_csv(
                                label_path, 
                                index=False, 
                                header=False, 
                                sep=' '
                            )
                    except ValueError as e:
                        print(f"Warning: Could not parse image number from {image_file}: {e}")
                        continue

def create_dataset_yaml() -> None:
    """Create YOLO dataset configuration file."""
    print("\n📄 Creating dataset YAML configuration...")
    
    # Dataset configuration
    data = {
        'path': str(BASE_DIR),
        'train': train_image_folders,
        'val': val_image_folders,
        'names': {0: 'Player', 1: 'Ball'}
    }
    
    # Add test set if available
    if test_image_folders:
        data['test'] = test_image_folders

    # Save dataset YAML
    yaml_path = BASE_DIR / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    
    print(f"✓ Dataset YAML created: {yaml_path}")
    
    # Save YAML path for reference
    yaml_path_file = Path(__file__).parent / 'yaml_path.txt'
    with open(yaml_path_file, 'w', encoding='utf-8') as f:
        f.write(str(yaml_path))
    
    print(f"✓ YAML path saved: {yaml_path_file}")
    
    # Print dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Training folders: {len(train_image_folders)}")
    print(f"  - Validation folders: {len(val_image_folders)}")
    print(f"  - Test folders: {len(test_image_folders)}")
    print(f"  - Classes: {data['names']}")


def main() -> None:
    """Main processing function."""
    try:
        process_soccernet_dataset()
        create_dataset_yaml()
        print("\n🎉 SoccerNet dataset processing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")


if __name__ == "__main__":
    main()
