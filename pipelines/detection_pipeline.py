"""Detection Pipeline for Soccer Analysis.

This module provides pipeline functions for running detection on videos,
images, and real-time streams. Core detection functions are in detect_players.py.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_detection import load_detection_model, get_detections
from player_annotations import AnnotatorManager
from pipelines.processing_pipeline import ProcessingPipeline
import cv2
import supervision as sv


class DetectionPipeline:
    """
    Modular pipeline for running object detection on various input sources.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize detection pipeline.
        
        Args:
            model_path: Path to YOLO detection model
        """
        self.model_path = model_path
        self.model = None
        self.annotator_manager = AnnotatorManager()
        self.processing_pipeline = ProcessingPipeline()
        
    def initialize_model(self):
        """
        Load the detection model.
        """
        if self.model is None:
            print("Loading detection model...")
            self.model = load_detection_model(self.model_path)
        return self.model
    
    def detect_frame_objects(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
        """
        Detect players, ball, and referees in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (player_detections, ball_detections, referee_detections)
        """
        if self.model is None:
            self.initialize_model()
        
        return get_detections(self.model, frame)
    
    def annotate_detections(self, frame: np.ndarray, player_detections: sv.Detections, 
                          ball_detections: sv.Detections, referee_detections: sv.Detections) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            frame: Input frame
            player_detections: Player detection results
            ball_detections: Ball detection results
            referee_detections: Referee detection results
            
        Returns:
            Annotated frame
        """
        return self.annotator_manager.annotate_all(frame, player_detections, ball_detections, referee_detections)
    
    def detect_in_video(self, video_path: str, output_path: str, frame_count: int = 300):
        """
        Detect and annotate objects in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_count: Number of frames to process
        """
        self.initialize_model()
            
        video_frames = self.processing_pipeline.read_video_frames(video_path, frame_count=frame_count)
        
        print("Processing frames with detection...")
        annotated_frames = []
        for i, frame in enumerate(video_frames):
            player_detections, ball_detections, referee_detections = self.detect_frame_objects(frame)
            annotated_frame = self.annotate_detections(frame, player_detections, ball_detections, referee_detections)
            annotated_frames.append(annotated_frame)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(video_frames)} frames")

        self.processing_pipeline.write_video_output(annotated_frames, output_path)
        print(f"Detection complete! Output saved to: {output_path}")

    def detect_realtime(self, video_path: str):
        """
        Run real-time object detection on a video stream.
        
        Args:
            video_path: Path to input video or camera index (0 for webcam)
        """
        self.initialize_model()

        print("Opening video stream...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        print("Starting real-time detection. Press 'q' to quit.")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            player_detections, ball_detections, referee_detections = self.detect_frame_objects(frame)
            annotated_frame = self.annotate_detections(frame, player_detections, ball_detections, referee_detections)
            
            cv2.imshow("Soccer Analysis - Real-time Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped.")


if __name__ == "__main__":
    from player_detection.detection_constants import model_path, test_video, test_video_output
    # Example usage - uncomment desired function
    
    pipeline = DetectionPipeline(model_path)
    # pipeline.detect_in_video(test_video, test_video_output, 300)
    pipeline.detect_realtime(test_video)