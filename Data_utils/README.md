# Data Utils Directory

This directory contains utility scripts for processing various datasets used in the Soccer Analysis project. These scripts are primarily for data preparation, format conversion, and preprocessing tasks across different data sources.

> **Note**: These scripts are included in the repository for reference purposes and documentation. They are not intended for regular usage in the main soccer analysis pipeline.

## 📁 Directory Structure

```
Data_utils/
├── External_Detections/     # External dataset processing utilities
├── SoccerNet_Detections/    # SoccerNet tracking data processing
├── SoccerNet_Keypoints/     # SoccerNet keypoint and calibration data
└── README.md               # This file
```

### Complete File Structure

```
Data_utils/
├── External_Detections/
│   ├── coco_to_yolo.py              # COCO to YOLO format conversion
│   ├── create_data_yaml.py          # YOLO dataset configuration generator
│   ├── merge_datasets.py            # Multi-dataset merger with class mapping
│   ├── slice_images.py              # SAHI-based image slicing
│   └── visualize_coco_dataset.py    # COCO dataset visualization tool
├── SoccerNet_Detections/
│   ├── constants.py                 # Dataset paths and configuration
│   ├── data_preprocessing.py        # MOT to YOLO format conversion
│   └── get_soccernet_data.py        # SoccerNet tracking data downloader
├── SoccerNet_Keypoints/
│   ├── constants.py                 # Dataset paths and field constants
│   ├── create_dataset_yaml.py       # YOLO pose dataset configuration
│   ├── downloader.py                # SoccerNet calibration data downloader
│   ├── get_pitch_object.py          # Green pitch area detection
│   ├── line_intersections.py        # Field keypoint calculation from lines
│   ├── process_images.py            # Unified processing pipeline
│   ├── transfer_json_files.py       # JSON file consolidation utility
│   └── temp.ipynb                   # Development/testing notebook
└── README.md                        # This documentation file
```

## 🔧 External_Detections

Scripts for processing external COCO format datasets and converting them to YOLO format.

### Scripts:

#### `coco_to_yolo.py`
Converts COCO format annotations to Ultralytics YOLO format, supporting both bounding boxes and segmentation masks.

**Features:**
- Handles bounding box and segmentation data
- Maps COCO category IDs to YOLO class indices
- Creates class mapping files
- Comprehensive error handling

**Usage:**
```bash
python coco_to_yolo.py path/to/coco_annotations.json path/to/output/labels --use_segments
```

#### `create_data_yaml.py`
Generates YOLO dataset configuration files for training.

**Features:**
- Creates data.yaml with proper paths
- Defines class names and indices
- Simple configuration generation

**Usage:**
```python
from create_data_yaml import create_dataset_yaml
create_dataset_yaml("path/to/dataset")
```

#### `merge_datasets.py`
Merges multiple COCO datasets with class mapping and image limits.

**Features:**
- Combines multiple COCO datasets
- Standardizes class names across datasets
- Limits images per dataset
- Handles different annotation formats
- Copies/renames image files

**Key functionality:**
- Maps various class names (player, Player, Team-A, etc.) to standard classes
- Applies per-dataset image limits to balance training data
- Creates unified dataset with consistent structure

#### `slice_images.py`
Uses SAHI (Slicing Aided Hyper Inference) to slice large images into smaller patches.

**Features:**
- Multiple slice sizes (160x160, 320x320, 640x640)
- Overlap-based slicing for better detection
- Batch processing for multiple datasets
- COCO format preservation

**Usage:**
```python
from slice_images import slice_datasets
slice_datasets(dataset_paths, output_folders)
```

#### `visualize_coco_dataset.py`
Visualization tool for COCO format datasets with bounding boxes and labels.

**Features:**
- Random image sampling
- Bounding box visualization
- Category labels with colors
- Save/display options
- Error handling for missing images

**Usage:**
```python
from visualize_coco_dataset import visualize_coco_dataset
visualize_coco_dataset(image_dir, annotation_file, num_samples=10)
```

## ⚽ SoccerNet_Detections

Scripts for processing SoccerNet tracking dataset and converting to YOLO format.

### Scripts:

#### `get_soccernet_data.py`
Downloads SoccerNet tracking dataset with multi-object tracking annotations.

**Features:**
- Downloads train/test/challenge splits
- Handles SoccerNet authentication
- Progress tracking and error handling

#### `data_preprocessing.py`
Processes SoccerNet tracking data and converts to YOLO format.

**Features:**
- Reads MOT format ground truth files
- Converts to normalized YOLO coordinates
- Identifies balls as smallest objects
- Creates train/val/test splits
- Generates dataset YAML configuration

**Processing pipeline:**
1. Reads SoccerNet MOT format annotations
2. Normalizes coordinates (assumes 1920x1080 images)
3. Converts from top-left corner to center coordinates
4. Identifies ball objects by minimum area
5. Creates YOLO format label files
6. Generates dataset configuration

## 🎯 SoccerNet_Keypoints

Scripts for processing SoccerNet calibration data and extracting field keypoints.

### Scripts:

#### `downloader.py`
Downloads SoccerNet calibration datasets including field line annotations.

**Features:**
- Downloads calibration and calibration-2023 tasks
- Handles multiple dataset splits
- Authentication management

#### `line_intersections.py`
Calculates field keypoints from SoccerNet line endpoints using geometric intersections.

**Key Features:**
- **LineIntersectionCalculator** class with comprehensive geometry methods
- Line-line intersection calculations
- Circle-line intersection for center circle points
- Point-to-line distance calculations
- Field keypoint mapping and visualization

**Keypoint Processing:**
- Converts SoccerNet line endpoints to field keypoints
- Handles various field line types (sidelines, penalty areas, goal areas, center circle)
- Supports visualization of calculated keypoints
- Normalizes coordinates for consistent processing

**Usage:**
```python
calculator = LineIntersectionCalculator()
calculator.load_soccernet_data("path/to/annotation.json")
keypoints, lines = calculator.calculate_field_keypoints()
```

#### `get_pitch_object.py`
Detects the complete green pitch area as a bounding box for object detection training.

**Key Features:**
- **PitchDetector** class using HSV color segmentation
- Green area detection with morphological operations
- Largest contour identification (pitch area)
- Normalized coordinate output
- Visualization with overlay masks

**Processing Pipeline:**
1. Convert image to HSV color space
2. Create mask for green colors (grass)
3. Apply morphological operations to clean mask
4. Find largest contour (pitch)
5. Calculate bounding box from extreme coordinates
6. Normalize coordinates to 0-1 range

#### `process_images.py`
Unified processing pipeline combining pitch detection with keypoint extraction.

**Key Features:**
- Combines pitch object detection and keypoint calculation
- Creates multiple output formats:
  - JSON annotations with complete metadata
  - Ultralytics YOLO pose format labels
  - Visualization images with annotations
  - Dataset configuration files

**Output Structure:**
```
unified_output/
├── annotations_json/        # Complete JSON annotations
├── processed_images/        # Visualization images
├── yolo_labels/            # Ultralytics YOLO format
└── dataset.yaml            # Training configuration
```

#### `transfer_json_files.py`
Utility to consolidate SoccerNet calibration JSON files from different splits.

**Features:**
- Consolidates train/test/valid JSON files
- Copy or move operations
- Batch processing support
- Verification and statistics

#### `create_dataset_yaml.py`
Creates Ultralytics YOLO dataset configuration for keypoint detection.

**Features:**
- Keypoints with (x, y, visibility) format
- Keypoint connections for visualization
- Named keypoint mapping
- Pose estimation configuration
- Dataset path configuration for training

#### `transfer_json_files.py`
Utility to consolidate SoccerNet calibration JSON files from different splits.

**Features:**
- Consolidates train/test/valid JSON files
- Copy or move operations
- Batch processing support
- Verification and statistics
- Unified directory structure creation

#### `temp.ipynb`
Development and testing Jupyter notebook for SoccerNet keypoints processing.

**Contents:**
- Interactive data exploration and visualization
- Line endpoint analysis and keypoint calculation testing
- Field keypoint visualization examples
- Development utilities for debugging processing pipeline

## 🚀 Usage Examples

### Convert External COCO Dataset
```bash
# Convert COCO to YOLO format
python External_Detections/coco_to_yolo.py dataset/train/_annotations.coco.json dataset/labels/train

# Create dataset YAML
python External_Detections/create_data_yaml.py
```

### Process SoccerNet Data
```bash
# Download SoccerNet data
python SoccerNet_Detections/get_soccernet_data.py

# Preprocess to YOLO format
python SoccerNet_Detections/data_preprocessing.py
```

### Extract Field Keypoints
```bash
# Download calibration data
python SoccerNet_Keypoints/downloader.py

# Process complete pipeline
python SoccerNet_Keypoints/process_images.py
```

## 📋 Dependencies

The scripts in this directory require various packages depending on functionality:

### Common Dependencies
- `numpy`
- `pandas`
- `opencv-python`
- `pathlib`
- `tqdm`
- `pyyaml`

### Dataset-Specific
- `pycocotools` (COCO processing)
- `sahi` (Image slicing)
- `matplotlib` (Visualization)
- `SoccerNet` (SoccerNet data access)

### Installation
```bash
pip install numpy pandas opencv-python tqdm pyyaml pycocotools sahi matplotlib
pip install SoccerNet  # For SoccerNet data access
```

## ⚙️ Configuration Files

Many scripts reference `constants.py` files for configuration:

### SoccerNet_Detections/constants.py
```python
dataset_dir = "path/to/soccernet/data"
soccernet_password = "your_password"
```

### SoccerNet_Keypoints/constants.py
Contains dataset paths, field dimensions, and keypoint mapping constants:
```python
dataset_dir = "path/to/soccernet/data"
calibration_dir = Path("path/to/calibration/data")
soccernet_password = "your_password"

# Field dimensions (FIFA standard)
FIELD_LENGTH = 105.0  # meters
FIELD_WIDTH = 68.0    # meters
GOAL_WIDTH = 7.32
CENTER_CIRCLE_RADIUS = 9.15

# Standard field keypoints in real-world coordinates
STANDARD_FIELD_KEYPOINTS = {...}  # Complete field keypoint mapping
KEYPOINT_MAPPING = {...}  # Detected to standard keypoint mapping
```

## 🔍 Data Formats

### COCO Format
Standard COCO JSON structure with images, annotations, and categories arrays.

### YOLO Format
Text files with space-separated values:
```
class_id center_x center_y width height
```

### YOLO Pose Format (Keypoints)
```
class_id center_x center_y width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
```
Where `kp_v` is visibility: 0=not visible, 1=occluded, 2=visible

### SoccerNet Format
JSON files with field line endpoints and metadata for geometric calculations.

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages listed above
2. **Path Issues**: Update constants.py files with correct data paths
3. **SoccerNet Access**: Ensure valid SoccerNet credentials
4. **Memory Issues**: Large datasets may require processing in batches

### File Not Found Errors
- Verify dataset paths in constants.py
- Check that datasets are downloaded and extracted properly
- Ensure consistent directory structure

### Format Conversion Issues
- Verify input format matches expected structure
- Check image and annotation file correspondence
- Validate coordinate ranges and formats

## 📚 Additional Notes

- **Reference Only**: These scripts are for documentation and reference
- **Dataset Specific**: Each subdirectory handles specific dataset types
- **Preprocessing Focus**: Primarily for data preparation, not real-time processing
- **YOLO Integration**: Most scripts output YOLO-compatible formats for training

For questions or issues with these utilities, refer to the main project documentation or the individual script docstrings for detailed usage information.