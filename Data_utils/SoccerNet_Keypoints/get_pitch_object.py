"""
A script to detect the pitch object from soccer field images by finding the green area.
The pitch object is defined as the complete green part in the image with bounding box
calculated from extreme coordinates (min/max x and y values).
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import cv2
import numpy as np
import json
import os
from typing import Tuple, Dict, Optional
from Data_utils.SoccerNet_Keypoints.constants import calibration_dir

class PitchDetector:
    """
    A class to detect the pitch object (green area) in soccer field images.
    """
    
    def __init__(self):
        # HSV color ranges for green grass detection
        # Lower bound: darker green
        self.lower_green = np.array([35, 40, 40])
        # Upper bound: lighter green  
        self.upper_green = np.array([85, 255, 255])
        
    def detect_green_area(self, image: np.ndarray) -> np.ndarray:
        """
        Detect green areas in the image using color segmentation.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where green areas are white (255) and others are black (0)
        """
        # Convert BGR to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green colors
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def find_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the largest contour in the binary mask.
        
        Args:
            mask: Binary mask image
            
        Returns:
            Largest contour or None if no contours found
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour (assumed to be the pitch)
        largest_contour = max(contours, key=cv2.contourArea)
        
        return largest_contour
    
    def get_pitch_bounding_box(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate the bounding box of the pitch from the largest green contour.
        
        Args:
            contour: Contour points of the pitch
            image_shape: (height, width) of the image
            
        Returns:
            Dictionary with normalized bounding box coordinates and metadata
        """
        if contour is None:
            return None
            
        height, width = image_shape[:2]
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate extreme coordinates
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
        
        # Normalize coordinates to 0-1 range
        x_min_norm = x_min / width
        y_min_norm = y_min / height
        x_max_norm = x_max / width
        y_max_norm = y_max / height
        
        # Calculate center and dimensions
        center_x = (x_min_norm + x_max_norm) / 2
        center_y = (y_min_norm + y_max_norm) / 2
        bbox_width = x_max_norm - x_min_norm
        bbox_height = y_max_norm - y_min_norm
        
        return {
            'class_id': 0,  # Pitch class
            'class_name': 'pitch',
            'center_x': center_x,
            'center_y': center_y,
            'width': bbox_width,
            'height': bbox_height,
            'x_min': x_min_norm,
            'y_min': y_min_norm,
            'x_max': x_max_norm,
            'y_max': y_max_norm,
            'area': bbox_width * bbox_height,
            'contour_area': cv2.contourArea(contour) / (width * height)  # Normalized contour area
        }
    
    def detect_pitch_from_image(self, image_path: str) -> Optional[Dict]:
        """
        Detect pitch object from an image file.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with pitch detection results or None if detection failed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
            
        # Detect green areas
        green_mask = self.detect_green_area(image)
        
        # Find largest contour (pitch)
        largest_contour = self.find_largest_contour(green_mask)
        
        if largest_contour is None:
            print(f"No green area detected in image: {image_path}")
            return None
            
        # Get bounding box
        pitch_bbox = self.get_pitch_bounding_box(largest_contour, image.shape)
        
        if pitch_bbox is None:
            return None
            
        return {
            'image_path': image_path,
            'image_shape': {'height': image.shape[0], 'width': image.shape[1]},
            'pitch_detection': pitch_bbox
        }
    
    def visualize_detection(self, image_path: str, detection_result: Dict, output_path: str = None):
        """
        Visualize the pitch detection on the image.
        
        Args:
            image_path: Path to the input image
            detection_result: Result from detect_pitch_from_image
            output_path: Path to save the annotated image (optional)
        """
        image = cv2.imread(image_path)
        if image is None:
            return
            
        height, width = image.shape[:2]
        pitch_data = detection_result['pitch_detection']
        
        # Convert normalized coordinates to pixel coordinates
        x_min = int(pitch_data['x_min'] * width)
        y_min = int(pitch_data['y_min'] * height)
        x_max = int(pitch_data['x_max'] * width)
        y_max = int(pitch_data['y_max'] * height)
        
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        # Add text annotation
        text = f"Pitch (Area: {pitch_data['area']:.3f})"
        cv2.putText(image, text, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show green mask overlay
        green_mask = self.detect_green_area(image)
        green_overlay = cv2.applyColorMap(green_mask, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(image, 0.7, green_overlay, 0.3, 0)
        
        if output_path:
            cv2.imwrite(output_path, combined)
            print(f"Annotated image saved to: {output_path}")
        else:
            cv2.imshow('Pitch Detection', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def process_soccernet_dataset():
    """
    Process the SoccerNet calibration dataset to extract pitch objects.
    """
    detector = PitchDetector()
    
    # Create output directories
    output_json_dir = calibration_dir / 'pitch_objects'
    output_image_dir = calibration_dir / 'pitch_detection_images'
    output_json_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset type
    for dataset_type in ['train', 'test', 'valid']:
        if not (calibration_dir / dataset_type).exists():
            continue
            
        print(f"Processing {dataset_type} dataset...")
        dataset_path = calibration_dir / dataset_type
        
        # Create output subdirectories
        json_output_dir = output_json_dir / dataset_type
        image_output_dir = output_image_dir / dataset_type
        json_output_dir.mkdir(parents=True, exist_ok=True)
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image in the dataset
        for image_file in os.listdir(dataset_path):
            if image_file.endswith('.jpg'):
                image_path = dataset_path / image_file
                
                # Detect pitch
                result = detector.detect_pitch_from_image(str(image_path))
                
                if result:
                    # Save detection result as JSON
                    json_filename = image_file.replace('.jpg', '_pitch.json')
                    json_path = json_output_dir / json_filename
                    
                    with open(json_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    # Save visualization
                    vis_filename = image_file.replace('.jpg', '_pitch_detection.jpg')
                    vis_path = image_output_dir / vis_filename
                    
                    detector.visualize_detection(str(image_path), result, str(vis_path))
                    
                    print(f"Processed: {image_file}")
                else:
                    print(f"Failed to detect pitch in: {image_file}")

if __name__ == "__main__":
    # Example usage
    detector = PitchDetector()
    
    # Process entire dataset
    process_soccernet_dataset()
    
    # Or process a single image
    # image_path = "path/to/your/image.jpg"
    # result = detector.detect_pitch_from_image(image_path)
    # if result:
    #     print(json.dumps(result, indent=2))
    #     detector.visualize_detection(image_path, result)