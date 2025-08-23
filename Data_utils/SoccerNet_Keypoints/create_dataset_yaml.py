"""
Create dataset.yaml configuration file for Ultralytics YOLO.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from Data_utils.SoccerNet_Keypoints.constants import calibration_dir

def create_yolo_dataset_config():
    """
    Create dataset.yaml configuration file for Ultralytics YOLO.
    """
    base_output_dir = calibration_dir / 'unified_output'
    
    yaml_content = f"""# SoccerNet Keypoints Dataset Configuration for Ultralytics YOLO
# Dataset structure for pose estimation with pitch object detection

# Dataset paths (relative to this yaml file)
path: {base_output_dir.absolute()}
train: yolo_labels/train
val: yolo_labels/valid
test: yolo_labels/test

# Number of keypoints
kpt_shape: [27, 3]  # 27 keypoints, each with (x, y, visibility)

# Class names
names:
  0: pitch

# Keypoint connections for visualization (optional)
kpt_connections:
  # Field boundary connections
  - [0, 1]   # Top left to big rect left top pt1
  - [1, 2]   # Big rect connections
  - [2, 3]
  - [3, 4]
  - [9, 0]   # Bottom left to top left
  - [16, 26] # Top right to bottom right
  - [11, 12] # Center line top to bottom
  - [13, 14] # Center circle connections

# Keypoint names (for reference)
keypoint_names:
  0: sideline_top_left
  1: big_rect_left_top_pt1
  2: big_rect_left_top_pt2
  3: big_rect_left_bottom_pt1
  4: big_rect_left_bottom_pt2
  5: small_rect_left_top_pt1
  6: small_rect_left_top_pt2
  7: small_rect_left_bottom_pt1
  8: small_rect_left_bottom_pt2
  9: sideline_bottom_left
  10: left_semicircle_right
  11: center_line_top
  12: center_line_bottom
  13: center_circle_top
  14: center_circle_bottom
  15: field_center
  16: sideline_top_right
  17: big_rect_right_top_pt1
  18: big_rect_right_top_pt2
  19: big_rect_right_bottom_pt1
  20: big_rect_right_bottom_pt2
  21: small_rect_right_top_pt1
  22: small_rect_right_top_pt2
  23: small_rect_right_bottom_pt1
  24: small_rect_right_bottom_pt2
  25: sideline_bottom_right
  26: right_semicircle_left

# Additional dataset information
download: false  # Set to true if you want to enable auto-download
nc: 1            # Number of classes
"""
    
    yaml_path = base_output_dir / 'dataset.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset configuration created at: {yaml_path}")
    
    # Also create a simple README
    readme_content = """# SoccerNet Keypoints Dataset

This dataset contains:

## Directory Structure:
- `annotations_json/`: Complete JSON annotations with pitch objects and keypoints
- `processed_images/`: Visualization images showing detections and keypoints
- `yolo_labels/`: Ultralytics YOLO format labels for training
- `dataset.yaml`: Configuration file for Ultralytics YOLO

## Usage:

### For Training with Ultralytics:
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### For Visualization:
The JSON files contain complete annotations that can be used for custom visualization and analysis.

## Format:
- **Pitch Object**: Single bounding box covering the complete green area (class 0)
- **Keypoints**: 27 field keypoints with (x, y, visibility) format
- **Coordinates**: All normalized to 0-1 range
"""
    
    readme_path = base_output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"README created at: {readme_path}")

if __name__ == "__main__":
    create_yolo_dataset_config()