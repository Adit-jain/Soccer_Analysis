"""Keypoint Detection Pipeline for Soccer Analysis.

This pipeline coordinates keypoint detection functionality and provides
high-level interfaces for keypoint-based analysis.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import os
import random

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from keypoint_detection import (
    load_keypoint_model, get_keypoint_detections, normalize_keypoints,
    denormalize_keypoints, filter_visible_keypoints, extract_field_corners,
    calculate_field_dimensions
)
from keypoint_detection import (
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
        """
        Initialize the keypoint detection model.
        """
        self.model = load_keypoint_model(self.model_path)
        print(f"Keypoint detection model loaded from: {self.model_path}")
        
    def detect_keypoints_in_frame(self, frame: np.ndarray, get_metadata: bool = False) -> Tuple[np.ndarray, Dict]:
        """Detect keypoints in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (keypoints, metadata) where keypoints has shape (N, 27, 3)
            and metadata contains detection information
        """
  
        detections, keypoints = get_keypoint_detections(self.model, frame)
        metadata = {}

        # Calculate Metadata - Extract field corners and calculate dimensions
        if get_metadata:
            corners = extract_field_corners(keypoints)
            dimensions = calculate_field_dimensions(corners)
            metadata = {
                'num_detections': len(detections) if detections is not None else 0,
                'field_corners': corners,
                'field_dimensions': dimensions,
                'image_shape': frame.shape[:2]
            }
        
        return keypoints, metadata
        
    def annotate_frame(self, frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = CONFIDENCE_THRESHOLD, 
                        draw_connections: bool = False, draw_labels: bool = True) -> np.ndarray:
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
        for kpts in filtered_keypoints:
            
            # Draw keypoint connections
            if draw_connections:
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
                    if draw_labels:
                        label = f"{kpt_idx}: {KEYPOINT_NAMES.get(kpt_idx, 'Unknown')}"
                        cv2.putText(annotated_frame, label, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        
        return annotated_frame
    
    def detect_in_images(self, image_dir: str, visualize: bool = False, samples: Optional[int] = 10) -> List[Tuple[np.ndarray, Dict]]:
        """Detect keypoints in a directory of images.
        
        Args:
            image_dir: Directory containing images
            visualize: Whether to display results interactively
            samples: Number of images to sample (None for all)
            
        Returns:
            List of keypoint detection results
        """
        if self.model is None:
            self.initialize_model()
            
        print("Loading images...")
        img_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                     if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if samples is not None and samples < len(img_paths):
            img_paths = random.sample(img_paths, samples)

        print("Running keypoint detection...")
        results = []
        
        for i, img_path in enumerate(img_paths):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            keypoints, metadata = self.detect_keypoints_in_frame(image)
            results.append((keypoints, metadata))
            
            if visualize:
                annotated_image = self.annotate_frame(image, keypoints)
                cv2.imshow(f"Keypoint Detection - Image {i+1}", annotated_image)
                print(f"Image {i+1}/{len(img_paths)} - Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return results
    
    def detect_realtime(self, video_path: str):
        """Run real-time keypoint detection on a video stream.
        
        Args:
            video_path: Path to input video or camera index (0 for webcam)
        """
        if self.model is None:
            self.initialize_model()

        print("Opening video stream...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        print("Starting real-time keypoint detection. Press 'q' to quit.")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            keypoints, metadata = self.detect_keypoints_in_frame(frame)
            annotated_frame = self.annotate_frame(frame, keypoints)
            
            cv2.imshow("Soccer Analysis - Real-time Keypoint Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time keypoint detection stopped.")

    def detect_in_video(self, video_path: str, output_path: str, frame_count: int = 300):
        """Detect and annotate keypoints in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_count: Number of frames to process
        """
        if self.model is None:
            self.initialize_model()
            
        print("Reading the video...")
        video_frames = read_video(video_path, frame_count=frame_count)
        
        print("Detecting keypoints...")
        annotated_frames = []
        
        for i, frame in enumerate(video_frames):
            keypoints, metadata = self.detect_keypoints_in_frame(frame)
            annotated_frame = self.annotate_frame(frame, keypoints)
            annotated_frames.append(annotated_frame)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(video_frames)} frames")

        print("Writing output video...")
        write_video(annotated_frames, output_path)
        print(f"Keypoint detection complete! Output saved to: {output_path}")


if __name__ == "__main__":
    from keypoint_detection.keypoint_constants import keypoint_model_path, test_video, test_video_output, test_images_path
    # Example usage - uncomment desired function
    
    pipeline = KeypointPipeline(keypoint_model_path)
    pipeline.detect_in_video(test_video, test_video_output, 300)
    # pipeline.detect_in_images(test_images_path, True, 10)
    # pipeline.detect_realtime(test_video)