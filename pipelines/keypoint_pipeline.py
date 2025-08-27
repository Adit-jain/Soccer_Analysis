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

from keypoint_detection import (
    load_keypoint_model, get_keypoint_detections, filter_visible_keypoints,
    extract_field_corners, calculate_field_dimensions
)
from keypoint_detection import KEYPOINT_NAMES, KEYPOINT_CONNECTIONS, CONFIDENCE_THRESHOLD
from player_annotations import AnnotatorManager
from pipelines.processing_pipeline import ProcessingPipeline


class KeypointPipeline:
    """Modular keypoint detection and analysis pipeline."""
    
    def __init__(self, model_path: str):
        """Initialize the keypoint detection pipeline.
        
        Args:
            model_path: Path to the YOLO keypoint detection model
        """
        self.model_path = model_path
        self.model = None
        self.annotator_manager = AnnotatorManager()
        self.processing_pipeline = ProcessingPipeline()
        
    def initialize_model(self):
        """
        Initialize the keypoint detection model.
        """
        if self.model is None:
            print("Loading keypoint detection model...")
            self.model = load_keypoint_model(self.model_path)
        return self.model
        
    def detect_keypoints_in_frame(self, frame: np.ndarray, get_metadata: bool = False) -> Tuple[np.ndarray, Dict]:
        """Detect keypoints in a single frame.
        
        Args:
            frame: Input frame as numpy array
            get_metadata: Whether to calculate additional metadata
            
        Returns:
            Tuple of (keypoints, metadata) where keypoints has shape (N, 29, 3)
            and metadata contains detection information
        """
        if self.model is None:
            self.initialize_model()
  
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
        
    def annotate_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, 
                          confidence_threshold: float = CONFIDENCE_THRESHOLD, 
                          draw_connections: bool = False, draw_labels: bool = True) -> np.ndarray:
        """Annotate frame with detected keypoints.
        
        Args:
            frame: Input frame to annotate
            keypoints: Detected keypoints array with shape (N, 29, 3)
            confidence_threshold: Minimum confidence to draw keypoint
            draw_connections: Whether to draw connections between keypoints
            draw_labels: Whether to draw keypoint labels
            
        Returns:
            Annotated frame
        """
        if keypoints is None or keypoints.size == 0:
            return frame
            
        # Filter visible keypoints
        filtered_keypoints = filter_visible_keypoints(keypoints, confidence_threshold)
        return self.annotator_manager.annotate_keypoints(
            frame, filtered_keypoints, confidence_threshold, 
            draw_vertices=True, draw_edges=draw_connections, draw_labels=draw_labels, 
            KEYPOINT_CONNECTIONS=KEYPOINT_CONNECTIONS, KEYPOINT_NAMES=KEYPOINT_NAMES
        )

    def detect_in_video(self, video_path: str, output_path: str, frame_count: int = 300):
        """Detect and annotate keypoints in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_count: Number of frames to process
        """
        self.initialize_model()
            
        video_frames = self.processing_pipeline.read_video_frames(video_path, frame_count=frame_count)
        
        print("Processing frames with keypoint detection...")
        annotated_frames = []
        
        for i, frame in enumerate(video_frames):
            keypoints, metadata = self.detect_keypoints_in_frame(frame)
            annotated_frame = self.annotate_keypoints(frame, keypoints)
            annotated_frames.append(annotated_frame)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(video_frames)} frames")

        self.processing_pipeline.write_video_output(annotated_frames, output_path)
        print(f"Keypoint detection complete! Output saved to: {output_path}")

    def detect_realtime(self, video_path: str):
        """Run real-time keypoint detection on a video stream.
        
        Args:
            video_path: Path to input video or camera index (0 for webcam)
        """
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
            annotated_frame = self.annotate_keypoints(frame, keypoints)
            
            cv2.imshow("Soccer Analysis - Real-time Keypoint Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time keypoint detection stopped.")


if __name__ == "__main__":
    from keypoint_detection.keypoint_constants import keypoint_model_path, test_video, test_video_output
    # Example usage - uncomment desired function
    
    pipeline = KeypointPipeline(keypoint_model_path)
    # pipeline.detect_in_video(test_video, test_video_output, 300)
    pipeline.detect_realtime(test_video)