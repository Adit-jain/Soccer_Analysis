"""Core Keypoint Detection Functions for Soccer Analysis.

This module provides core functionality for detecting soccer field keypoints
using YOLO pose estimation models and geometric calculations.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from ultralytics import YOLO
import supervision as sv

# ================================================
# Core Detection Functions
# ================================================

def load_keypoint_model(model_path: str) -> YOLO:
    """Load and return a YOLO pose estimation model for keypoint detection.
    
    Args:
        model_path: Path to the YOLO pose model file
        
    Returns:
        YOLO model instance configured for pose estimation
    """
    model = YOLO(model_path)
    return model


def detect_keypoints_in_frames(model: YOLO, frames) -> List:
    """Detect keypoints in video frames using YOLO pose model.
    
    Args:
        model: Loaded YOLO pose model
        frames: Video frames or single frame
        
    Returns:
        Detection results from YOLO pose model containing keypoints
    """
    return model(frames)


def get_keypoint_detections(keypoint_model: YOLO, frame: np.ndarray) -> Tuple[sv.Detections, np.ndarray]:
    """Get keypoint detections and extract keypoint coordinates.
    
    Args:
        keypoint_model: Loaded YOLO pose model
        frame: Input frame as numpy array
        
    Returns:
        Tuple of (detections, keypoints) where keypoints is array of shape (N, 27, 3)
        for N detections with 27 keypoints each having (x, y, visibility)
    """
    results = detect_keypoints_in_frames(keypoint_model, frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Extract keypoints if available
    keypoints = None
    if hasattr(results, 'keypoints') and results.keypoints is not None:
        keypoints = results.keypoints.data.cpu().numpy()  # Shape: (N, 27, 3)
    
    return detections, keypoints


def normalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Normalize keypoint coordinates to 0-1 range.
    
    Args:
        keypoints: Array of keypoints with shape (N, 27, 3)
        image_width: Width of the source image
        image_height: Height of the source image
        
    Returns:
        Normalized keypoints array with coordinates in 0-1 range
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints
        
    normalized_keypoints = keypoints.copy()
    normalized_keypoints[:, :, 0] /= image_width   # Normalize x coordinates
    normalized_keypoints[:, :, 1] /= image_height  # Normalize y coordinates
    # Visibility values (index 2) remain unchanged
    
    return normalized_keypoints


def denormalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Denormalize keypoint coordinates from 0-1 range to image coordinates.
    
    Args:
        keypoints: Array of normalized keypoints with shape (N, 27, 3)
        image_width: Width of the target image
        image_height: Height of the target image
        
    Returns:
        Denormalized keypoints array with pixel coordinates
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints
        
    denormalized_keypoints = keypoints.copy()
    denormalized_keypoints[:, :, 0] *= image_width   # Denormalize x coordinates
    denormalized_keypoints[:, :, 1] *= image_height  # Denormalize y coordinates
    # Visibility values remain unchanged
    
    return denormalized_keypoints


def filter_visible_keypoints(keypoints: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
    """Filter keypoints based on visibility confidence.
    
    Args:
        keypoints: Array of keypoints with shape (N, 27, 3)
        confidence_threshold: Minimum confidence for a keypoint to be considered visible
        
    Returns:
        Filtered keypoints with low-confidence points set to (0, 0, 0)
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints
        
    filtered_keypoints = keypoints.copy()
    
    # Set invisible keypoints to (0, 0, 0)
    invisible_mask = keypoints[:, :, 2] < confidence_threshold
    filtered_keypoints[invisible_mask] = 0
    
    return filtered_keypoints


def extract_field_corners(keypoints: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Extract the four corner points of the soccer field from detected keypoints.
    
    Args:
        keypoints: Array of keypoints with shape (N, 27, 3)
        
    Returns:
        Dictionary containing corner coordinates:
        {'top_left': (x, y), 'top_right': (x, y), 
         'bottom_left': (x, y), 'bottom_right': (x, y)}
    """
    if keypoints is None or keypoints.size == 0:
        return {'top_left': (0, 0), 'top_right': (0, 0), 
                'bottom_left': (0, 0), 'bottom_right': (0, 0)}
    
    # Assuming first detection and using specific keypoint indices for field corners
    # Based on the 27-keypoint model structure from the dataset
    corners = {}
    
    if keypoints.shape[0] > 0:
        kpts = keypoints[0]  # First detection
        
        # Extract corner keypoints (indices based on dataset structure)
        # These indices correspond to the field boundary points
        corners['top_left'] = (float(kpts[0, 0]), float(kpts[0, 1])) if kpts[0, 2] > 0 else (0, 0)
        corners['top_right'] = (float(kpts[16, 0]), float(kpts[16, 1])) if kpts[16, 2] > 0 else (0, 0)
        corners['bottom_left'] = (float(kpts[9, 0]), float(kpts[9, 1])) if kpts[9, 2] > 0 else (0, 0)
        corners['bottom_right'] = (float(kpts[25, 0]), float(kpts[25, 1])) if kpts[25, 2] > 0 else (0, 0)
    
    return corners


def calculate_field_dimensions(corners: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Calculate field dimensions from corner keypoints.
    
    Args:
        corners: Dictionary of corner coordinates
        
    Returns:
        Dictionary containing field measurements:
        {'width': field_width, 'height': field_height, 'area': field_area}
    """
    top_left = corners['top_left']
    top_right = corners['top_right']
    bottom_left = corners['bottom_left']
    bottom_right = corners['bottom_right']
    
    # Calculate field dimensions
    width = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
    height = np.sqrt((bottom_left[0] - top_left[0])**2 + (bottom_left[1] - top_left[1])**2)
    area = width * height
    
    return {
        'width': float(width),
        'height': float(height), 
        'area': float(area)
    }