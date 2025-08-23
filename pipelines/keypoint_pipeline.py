"""Keypoint Detection Pipeline for Soccer Analysis.

This pipeline coordinates keypoint detection functionality and provides
high-level interfaces for keypoint-based analysis.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from keypoint_detection.detect_keypoints import (
    load_keypoint_model, get_keypoint_detections, normalize_keypoints,
    denormalize_keypoints, filter_visible_keypoints, extract_field_corners,
    calculate_field_dimensions
)
from keypoint_detection.keypoint_constants import (
    KEYPOINT_NAMES, KEYPOINT_CONNECTIONS, FIELD_CORNERS, 
    KEYPOINT_COLOR, CONNECTION_COLOR, FIELD_CORNER_COLOR, TEXT_COLOR,
    CONFIDENCE_THRESHOLD, NUM_KEYPOINTS
)
from utils.vid_utils import read_video, write_video


class KeypointPipeline:
    """Complete keypoint detection and analysis pipeline."""
    
    def __init__(self, model_path: str):
        """Initialize the keypoint detection pipeline.
        
        Args:
            model_path: Path to the YOLO keypoint detection model
        """
        self.model_path = model_path
        self.model = None
        
    def initialize_model(self):
        """Initialize the keypoint detection model."""
        self.model = load_keypoint_model(self.model_path)
        print(f"Keypoint detection model loaded from: {self.model_path}")
        
    def detect_keypoints_in_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect keypoints in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (keypoints, metadata) where keypoints has shape (N, 27, 3)
            and metadata contains detection information
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
            
        detections, keypoints = get_keypoint_detections(self.model, frame)
        
        # Extract field corners and calculate dimensions
        corners = extract_field_corners(keypoints)
        dimensions = calculate_field_dimensions(corners)
        
        metadata = {
            'num_detections': len(detections) if detections is not None else 0,
            'field_corners': corners,
            'field_dimensions': dimensions,
            'image_shape': frame.shape[:2]
        }
        
        return keypoints, metadata
        
    def detect_keypoints_in_video(self, video_path: str, output_path: Optional[str] = None) -> List[Tuple[np.ndarray, Dict]]:
        """Detect keypoints in video frames.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated output video
            
        Returns:
            List of (keypoints, metadata) tuples for each frame
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
            
        frames = read_video(video_path)
        results = []
        annotated_frames = []
        
        print(f"Processing {len(frames)} frames...")
        
        for i, frame in enumerate(frames):
            keypoints, metadata = self.detect_keypoints_in_frame(frame)
            results.append((keypoints, metadata))
            
            # Create annotated frame if output is requested
            if output_path:
                annotated_frame = self.annotate_frame(frame, keypoints)
                annotated_frames.append(annotated_frame)
                
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(frames)} frames")
        
        # Write annotated video if requested
        if output_path and annotated_frames:
            write_video(annotated_frames, output_path, fps=30)
            print(f"Annotated video saved to: {output_path}")
            
        return results
        
    def annotate_frame(self, frame: np.ndarray, keypoints: np.ndarray, 
                      confidence_threshold: float = CONFIDENCE_THRESHOLD) -> np.ndarray:
        """Annotate frame with detected keypoints.
        
        Args:
            frame: Input frame to annotate
            keypoints: Detected keypoints array with shape (N, 27, 3)
            confidence_threshold: Minimum confidence to draw keypoint
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        if keypoints is None or keypoints.size == 0:
            return annotated_frame
            
        # Filter visible keypoints
        filtered_keypoints = filter_visible_keypoints(keypoints, confidence_threshold)
        
        # Draw keypoints and connections for each detection
        for detection_idx in range(filtered_keypoints.shape[0]):
            kpts = filtered_keypoints[detection_idx]
            
            # Draw keypoint connections
            for connection in KEYPOINT_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                pt1 = kpts[pt1_idx]
                pt2 = kpts[pt2_idx]
                
                # Only draw if both points are visible
                if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                    cv2.line(annotated_frame, 
                           (int(pt1[0]), int(pt1[1])),
                           (int(pt2[0]), int(pt2[1])),
                           CONNECTION_COLOR, 2)
            
            # Draw keypoints
            for kpt_idx, kpt in enumerate(kpts):
                if kpt[2] > confidence_threshold:  # Check visibility
                    x, y = int(kpt[0]), int(kpt[1])
                    
                    # Use different color for field corners
                    color = FIELD_CORNER_COLOR if kpt_idx in FIELD_CORNERS.values() else KEYPOINT_COLOR
                    
                    # Draw keypoint circle
                    cv2.circle(annotated_frame, (x, y), 5, color, -1)
                    
                    # Draw keypoint label
                    label = f"{kpt_idx}: {KEYPOINT_NAMES.get(kpt_idx, 'Unknown')}"
                    cv2.putText(annotated_frame, label, (x + 10, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        
        return annotated_frame
        
    def extract_field_transformation(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Extract transformation matrix from field keypoints to standard view.
        
        Args:
            keypoints: Detected keypoints array with shape (N, 27, 3)
            
        Returns:
            Transformation matrix (3x3) or None if insufficient keypoints
        """
        if keypoints is None or keypoints.size == 0:
            return None
            
        corners = extract_field_corners(keypoints)
        
        # Check if we have all four corners
        valid_corners = [pt for pt in corners.values() if pt != (0, 0)]
        if len(valid_corners) < 4:
            return None
            
        # Define source points (detected corners)
        src_points = np.array([
            corners['top_left'],
            corners['top_right'], 
            corners['bottom_right'],
            corners['bottom_left']
        ], dtype=np.float32)
        
        # Define destination points (standard field view)
        dst_points = np.array([
            [0, 0],
            [1000, 0],
            [1000, 680],
            [0, 680]
        ], dtype=np.float32)
        
        # Calculate perspective transformation
        transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return transformation_matrix
        
    def apply_field_transformation(self, frame: np.ndarray, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Apply field transformation to get top-down view.
        
        Args:
            frame: Input frame
            keypoints: Detected keypoints
            
        Returns:
            Transformed frame with top-down field view or None if transformation fails
        """
        transformation_matrix = self.extract_field_transformation(keypoints)
        
        if transformation_matrix is None:
            return None
            
        # Apply perspective transformation
        transformed_frame = cv2.warpPerspective(frame, transformation_matrix, (1000, 680))
        
        return transformed_frame
        
    def analyze_field_coverage(self, keypoints: np.ndarray, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> Dict:
        """Analyze field coverage based on visible keypoints.
        
        Args:
            keypoints: Detected keypoints array
            confidence_threshold: Minimum confidence for visible keypoint
            
        Returns:
            Dictionary with coverage analysis
        """
        if keypoints is None or keypoints.size == 0:
            return {'coverage_percentage': 0, 'visible_keypoints': 0, 'missing_regions': list(KEYPOINT_NAMES.keys())}
            
        filtered_keypoints = filter_visible_keypoints(keypoints, confidence_threshold)
        
        # Count visible keypoints
        visible_count = 0
        visible_indices = []
        
        for detection_idx in range(filtered_keypoints.shape[0]):
            kpts = filtered_keypoints[detection_idx]
            for kpt_idx, kpt in enumerate(kpts):
                if kpt[2] > confidence_threshold:
                    visible_count += 1
                    visible_indices.append(kpt_idx)
        
        coverage_percentage = (visible_count / (NUM_KEYPOINTS * filtered_keypoints.shape[0])) * 100
        missing_indices = [idx for idx in range(NUM_KEYPOINTS) if idx not in visible_indices]
        
        return {
            'coverage_percentage': coverage_percentage,
            'visible_keypoints': visible_count,
            'total_possible_keypoints': NUM_KEYPOINTS * filtered_keypoints.shape[0],
            'missing_keypoint_indices': missing_indices,
            'missing_keypoint_names': [KEYPOINT_NAMES[idx] for idx in missing_indices]
        }


# ================================================
# Convenience Functions for Pipeline Usage
# ================================================

def detect_keypoints_in_image(model_path: str, image_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """Detect keypoints in a single image.
    
    Args:
        model_path: Path to keypoint detection model
        image_path: Path to input image
        output_path: Optional path to save annotated image
        
    Returns:
        Tuple of (keypoints, metadata)
    """
    pipeline = KeypointPipeline(model_path)
    pipeline.initialize_model()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from: {image_path}")
    
    # Detect keypoints
    keypoints, metadata = pipeline.detect_keypoints_in_frame(image)
    
    # Save annotated image if requested
    if output_path:
        annotated_image = pipeline.annotate_frame(image, keypoints)
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")
    
    return keypoints, metadata


def detect_keypoints_in_video_batch(model_path: str, video_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, List[Tuple[np.ndarray, Dict]]]:
    """Process multiple videos for keypoint detection.
    
    Args:
        model_path: Path to keypoint detection model
        video_paths: List of input video paths
        output_dir: Optional directory to save annotated videos
        
    Returns:
        Dictionary mapping video paths to their detection results
    """
    pipeline = KeypointPipeline(model_path)
    pipeline.initialize_model()
    
    results = {}
    
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            video_name = Path(video_path).stem
            output_path = str(output_dir / f"{video_name}_keypoints.mp4")
        
        video_results = pipeline.detect_keypoints_in_video(video_path, output_path)
        results[video_path] = video_results
    
    return results


if __name__ == "__main__":
    # Example usage
    from keypoint_detection.keypoint_constants import keypoint_model_path, test_images_path
    
    # Test with single image
    test_image = Path(test_images_path) / "00000.jpg"
    if test_image.exists():
        print("Testing keypoint detection on single image...")
        keypoints, metadata = detect_keypoints_in_image(
            str(keypoint_model_path), 
            str(test_image),
            "test_keypoint_output.jpg"
        )
        print(f"Detection metadata: {metadata}")
    else:
        print(f"Test image not found: {test_image}")