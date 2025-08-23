"""Unified SoccerNet keypoints and pitch object processing.

This script combines pitch object detection and keypoint extraction
into a single processing pipeline, creating multiple output formats
for different ML frameworks and use cases.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import tqdm

# Project setup
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from Data_utils.SoccerNet_Keypoints.constants import calibration_dir
from Data_utils.SoccerNet_Keypoints.get_pitch_object import PitchDetector
from Data_utils.SoccerNet_Keypoints.line_intersections import LineIntersectionCalculator

def create_ultralytics_annotation(
    pitch_data: Dict, 
    keypoints: Dict, 
    image_shape: Tuple[int, int]
) -> str:
    """Create Ultralytics YOLO pose format annotation string.
    
    Format: <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
    
    Args:
        pitch_data: Pitch bounding box data with center_x, center_y, width, height
        keypoints: Dictionary mapping keypoint names to (x, y) coordinates
        image_shape: (height, width) of the image
        
    Returns:
        Formatted annotation string for Ultralytics YOLO
    """
    # Pitch object (class 0)
    annotation_parts = [
        "0",  # Class index for pitch
        f"{pitch_data['center_x']:.6f}",
        f"{pitch_data['center_y']:.6f}", 
        f"{pitch_data['width']:.6f}",
        f"{pitch_data['height']:.6f}"
    ]
    
    # Add keypoints in a consistent order
    keypoint_order = [
        '1_sideline_top_left', '2_big_rect_left_top_pt1', '3_big_rect_left_top_pt2',
        '4_big_rect_left_bottom_pt1', '5_big_rect_left_bottom_pt2', '6_small_rect_left_top_pt1',
        '7_small_rect_left_top_pt2', '8_small_rect_left_bottom_pt1', '9_small_rect_left_bottom_pt2',
        '10_sideline_bottom_left', '11_left_semicircle_right', '12_center_line_top',
        '13_center_line_bottom', '14_center_circle_top', '15_center_circle_bottom',
        '16_field_center', '17_sideline_top_right', '18_big_rect_right_top_pt1',
        '19_big_rect_right_top_pt2', '20_big_rect_right_bottom_pt1', '21_big_rect_right_bottom_pt2',
        '22_small_rect_right_top_pt1', '23_small_rect_right_top_pt2', '24_small_rect_right_bottom_pt1',
        '25_small_rect_right_bottom_pt2', '26_sideline_bottom_right', '27_right_semicircle_left'
    ]
    
    # Add keypoints with visibility flag (2 = visible, 0 = not visible)
    for kp_name in keypoint_order:
        if kp_name in keypoints:
            x, y = keypoints[kp_name]
            annotation_parts.extend([f"{x:.6f}", f"{y:.6f}", "2"])  # 2 = visible
        else:
            annotation_parts.extend(["0.0", "0.0", "0"])  # 0 = not visible
    
    return " ".join(annotation_parts)

def create_unified_visualization(
    image_path: str, 
    pitch_data: Dict, 
    keypoints: Dict,
    lines: Dict, 
    output_path: str
) -> None:
    """Create comprehensive visualization showing pitch detection and keypoints.
    
    Args:
        image_path: Path to source image
        pitch_data: Pitch bounding box information
        keypoints: Dictionary of calculated keypoints
        lines: Original line data from SoccerNet
        output_path: Path to save annotated image
    """
    image = cv2.imread(image_path)
    if image is None:
        return
        
    height, width = image.shape[:2]
    
    # Draw pitch bounding box
    x_min = int(pitch_data['x_min'] * width)
    y_min = int(pitch_data['y_min'] * height)
    x_max = int(pitch_data['x_max'] * width)
    y_max = int(pitch_data['y_max'] * height)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.putText(image, f"Pitch", (x_min, y_min - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw original lines in green
    if lines:
        for line_name, line_points in lines.items():
            if (len(line_points) >= 2 and 
                line_name not in ['Circle central', 'Circle left', 'Circle right']):
                pt1 = (
                    int(line_points[0]['x'] * width), 
                    int(line_points[0]['y'] * height)
                )
                pt2 = (
                    int(line_points[1]['x'] * width), 
                    int(line_points[1]['y'] * height)
                )
                cv2.line(image, pt1, pt2, (0, 150, 0), 1)
        
        # Draw circle points
        for circle_name in ['Circle central', 'Circle left', 'Circle right']:
            if circle_name in lines:
                for point in lines[circle_name]:
                    pt = (int(point['x'] * width), int(point['y'] * height))
                    cv2.circle(image, pt, 2, (0, 150, 0), -1)
    
    # Draw calculated keypoints in red
    for i, (keypoint_name, (x, y)) in enumerate(keypoints.items()):
        pt = (int(x * width), int(y * height))
        cv2.circle(image, pt, 6, (0, 0, 255), -1)
        # Add keypoint number for identification
        kp_num = keypoint_name.split('_')[0]
        cv2.putText(image, kp_num, (pt[0] + 8, pt[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, image)

def process_unified_soccernet_dataset() -> None:
    """Unified SoccerNet processing pipeline.
    
    Combines pitch object detection with keypoint extraction to create:
    - JSON annotations with complete metadata
    - Ultralytics YOLO format labels
    - Visualization images with annotations
    - Dataset configuration files
    """
    print("📎 Starting unified SoccerNet dataset processing...")
    
    # Initialize processors
    calculator = LineIntersectionCalculator()
    pitch_detector = PitchDetector()
    
    # Create output directory structure
    base_output_dir = calibration_dir / 'unified_output'
    json_dir = base_output_dir / 'annotations_json'  # Combined JSON annotations
    images_dir = base_output_dir / 'processed_images'  # Visualization images
    yolo_labels_dir = base_output_dir / 'yolo_labels'  # Ultralytics format
    
    # Create all output directories
    for dir_path in [base_output_dir, json_dir, images_dir, yolo_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset type
    for dataset_type in ['train', 'test', 'valid']:
        if not (calibration_dir / dataset_type).exists():
            continue
            
        print(f"\n📁 Processing {dataset_type} dataset...")
        annotations_path = calibration_dir / 'soccernet_calibration_annotations' / dataset_type
        images_path = calibration_dir / dataset_type
        
        if not annotations_path.exists():
            print(f"  Warning: Annotations path not found: {annotations_path}")
            continue
        
        # Create subdirectories for each split
        for output_dir in [json_dir, images_dir, yolo_labels_dir]:
            (output_dir / dataset_type).mkdir(parents=True, exist_ok=True)
        
        # Process each image in the dataset
        json_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
        print(f"  Found {len(json_files)} annotation files")
        
        for json_file in tqdm.tqdm(json_files, desc=f"Processing {dataset_type}"):
            json_path = annotations_path / json_file
            image_path = images_path / json_file.replace('.json', '.jpg')
            base_name = json_file.replace('.json', '')
                
                if not image_path.exists():
                    print(f"  Warning: Image not found: {image_path}")
                    continue
                
                try:
                    # 1. Calculate keypoints from lines
                    calculator.load_soccernet_data(str(json_path))
                    keypoints, lines = calculator.calculate_field_keypoints()
                    
                    # 2. Detect pitch object
                    pitch_result = pitch_detector.detect_pitch_from_image(str(image_path))
                    
                    if not pitch_result:
                        print(f"  Warning: Failed to detect pitch in: {image_path.name}")
                        continue
                except Exception as e:
                    print(f"  Error processing {json_file}: {e}")
                    continue
                
                pitch_data = pitch_result['pitch_detection']
                image_shape = (pitch_result['image_shape']['height'], 
                              pitch_result['image_shape']['width'])
                
                # 3. Create unified JSON annotation
                unified_annotation = {
                    'image_info': {
                        'file_name': image_path.name,
                        'path': str(image_path),
                        'width': image_shape[1],
                        'height': image_shape[0]
                    },
                    'pitch_object': pitch_data,
                    'keypoints': keypoints,
                    'original_lines': lines,
                    'dataset_split': dataset_type,
                    'total_keypoints': len(keypoints),
                    'annotation_format': 'SoccerNet_unified_v1'
                }
                
                # Save unified JSON
                json_output_path = json_dir / dataset_type / f"{base_name}.json"
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(unified_annotation, f, indent=2)
                
                # 4. Create Ultralytics YOLO format label
                yolo_annotation = create_ultralytics_annotation(
                    pitch_data, keypoints, image_shape
                )
                yolo_output_path = yolo_labels_dir / dataset_type / f"{base_name}.txt"
                with open(yolo_output_path, 'w', encoding='utf-8') as f:
                    f.write(yolo_annotation + '\n')
                
                # 5. Create unified visualization
                vis_output_path = images_dir / dataset_type / f"{base_name}_annotated.jpg"
                create_unified_visualization(
                    str(image_path), pitch_data, keypoints, 
                    lines, str(vis_output_path)
                )
                
    print(f"\n🎉 Processing complete! Output directories:")
    print(f"  📄 JSON annotations: {json_dir}")
    print(f"  🎨 Processed images: {images_dir}")
    print(f"  🏷️ YOLO labels: {yolo_labels_dir}")
    
    # Create dataset configuration
    create_yolo_dataset_config(base_output_dir)
    print(f"\n✅ Unified SoccerNet processing pipeline completed!")

def create_yolo_dataset_config(base_output_dir: Path) -> None:
    """Create dataset.yaml configuration file for Ultralytics YOLO.
    
    Args:
        base_output_dir: Base directory where dataset.yaml will be saved
    """
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
  # Add more connections as needed for visualization

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
"""
    
    yaml_path = base_output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  📄 Dataset configuration saved: {yaml_path}")

def main() -> None:
    """Main function to execute the unified processing pipeline."""
    try:
        process_unified_soccernet_dataset()
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()