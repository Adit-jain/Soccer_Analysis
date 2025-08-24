"""Detection Pipeline for Soccer Analysis.

This module provides pipeline functions for running detection on videos,
images, and real-time streams. Core detection functions are in detect_players.py.
"""

import sys
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_detection import load_detection_model, detect_objects_in_frames
from utils import read_video, write_video
import cv2
import random
import os
import numpy as np


def draw_bounding_boxes(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, 
                       confs: np.ndarray, class_names: dict) -> np.ndarray:
    """Draw bounding boxes with labels on the image.
    
    Args:
        image: Input image
        boxes: Bounding boxes in xywh format
        classes: Class IDs
        confs: Confidence scores
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        Image with drawn bounding boxes
    """
    annotated_image = image.copy()
    
    for box, cls, conf in zip(boxes, classes, confs):
        x, y, w, h = box
        # Convert from center format to corner format
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        class_name = class_names[int(cls)]
        
        # Draw rectangle and label
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{class_name} {conf:.2f}", 
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image


class DetectionPipeline:
    """
    Pipeline for running object detection on various input sources.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize detection pipeline.
        
        Args:
            model_path: Path to YOLO detection model
        """
        self.model_path = model_path
        self.model = None
        
    def initialize_model(self):
        """
        Load the detection model.
        """
        print("Loading detection model...")
        self.model = load_detection_model(self.model_path)
        return self.model
    
    def detect_in_video(self, video_path: str, output_path: str, frame_count: int = 300):
        """
        Detect and annotate objects in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_count: Number of frames to process
        """
        if self.model is None:
            self.initialize_model()
            
        print("Reading the video...")
        video_frames = read_video(video_path, frame_count=frame_count)
        
        print("Detecting objects...")
        results = detect_objects_in_frames(self.model, video_frames)
        
        print("Annotating frames...")
        annotated_frames = []
        for index, result in enumerate(results):
            frame = video_frames[index]
            
            if result.boxes is not None:
                class_names = result.names
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                frame = draw_bounding_boxes(frame, boxes, classes, confs, class_names)
            
            annotated_frames.append(frame)

        print("Writing output video...")
        write_video(annotated_frames, output_path)
        print(f"Detection complete! Output saved to: {output_path}")

    def detect_in_images(self, image_dir: str, model_2_path: Optional[str] = None, 
                        visualize: bool = False, samples: Optional[int] = 10) -> List:
        """
        Detect objects in a directory of images.
        
        Args:
            image_dir: Directory containing images
            model_2_path: Optional path to secondary model for comparison
            visualize: Whether to display results interactively
            samples: Number of images to sample (None for all)
            
        Returns:
            List of detection results for each model
        """
        if self.model is None:
            self.initialize_model()
            
        print("Loading images...")
        img_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                     if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if samples is not None and samples < len(img_paths):
            img_paths = random.sample(img_paths, samples)

        print("Loading detection models...")
        models = [self.model]
        
        if model_2_path is not None:
            models.append(load_detection_model(model_2_path))

        print("Running detection...")
        results = [detect_objects_in_frames(model, img_paths) for model in models]

        if visualize:
            print("Displaying results...")
            for image_idx in range(len(results[0])):
                for model_idx, model_results in enumerate(results):
                    print(f"Model {model_idx + 1} - Image {image_idx + 1}")
                    model_results[image_idx].show()
                input("Press Enter to continue...")

        return results

    def detect_realtime(self, video_path: str):
        """
        Run real-time object detection on a video stream.
        
        Args:
            video_path: Path to input video or camera index (0 for webcam)
        """
        if self.model is None:
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
                
            results = detect_objects_in_frames(self.model, frame)
            annotated_frame = results[0].plot()
            
            cv2.imshow("Soccer Analysis - Real-time Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped.")


if __name__ == "__main__":
    from player_detection.detection_constants import model_path, test_video, test_video_output, test_image_dir
    # Example usage - uncomment desired function
    
    pipeline = DetectionPipeline(model_path)
    # pipeline.detect_in_video(test_video, test_video_output, 300)
    # pipeline.detect_in_images(test_image_dir, None, True, 10)
    pipeline.detect_realtime(test_video)