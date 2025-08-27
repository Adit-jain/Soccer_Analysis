"""Tactical Analysis Pipeline for Soccer Analysis.

This pipeline coordinates tactical analysis functionality by combining
keypoint detection, player/ball/referee detection, and field coordinate transformations.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from keypoint_detection import (
    load_keypoint_model, get_keypoint_detections, filter_visible_keypoints
)
from keypoint_detection import KEYPOINT_NAMES, CONFIDENCE_THRESHOLD
from player_detection import load_detection_model, get_detections
from tactical_analysis.homography import HomographyTransformer
from player_annotations import AnnotatorManager
from utils.vid_utils import read_video, write_video
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv


class TacticalPipeline:
    """Complete tactical analysis pipeline for soccer field coordinate transformations."""
    
    def __init__(self, keypoint_model_path: str, detection_model_path: str):
        """Initialize the tactical analysis pipeline.
        
        Args:
            keypoint_model_path: Path to the YOLO keypoint detection model
            detection_model_path: Path to the YOLO detection model
        """
        self.keypoint_model_path = keypoint_model_path
        self.detection_model_path = detection_model_path
        self.keypoint_model = None
        self.detection_model = None
        self.homography_transformer = HomographyTransformer()
        self.annotator_manager = AnnotatorManager()
        self.pitch_config = SoccerPitchConfiguration()
        
    def initialize_models(self):
        """Initialize keypoint and detection models."""
        print("Loading keypoint detection model...")
        self.keypoint_model = load_keypoint_model(self.keypoint_model_path)
        
        print("Loading player detection model...")
        self.detection_model = load_detection_model(self.detection_model_path)
        
        print("Models initialized successfully")
    
    def detect_frame_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Detect keypoints in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Detected keypoints array with shape (N, 29, 3)
        """
        detections, keypoints = get_keypoint_detections(self.keypoint_model, frame)
        return keypoints
    
    def detect_frame_objects(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
        """Detect players, ball, and referees in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (player_detections, ball_detections, referee_detections)
        """
        player_detections, ball_detections, referee_detections = get_detections(
            self.detection_model, frame
        )
        return player_detections, ball_detections, referee_detections
    
    def transform_keypoints_to_pitch(self, detected_keypoints: np.ndarray) -> ViewTransformer:
        """Transform frame keypoints to pitch coordinate system.
        
        Args:
            detected_keypoints: Array of shape (1, 29, 3) with [x, y, confidence]
            
        Returns:
            ViewTransformer object for frame-to-pitch transformation
        """
        return self.homography_transformer.transform_to_pitch_keypoints(detected_keypoints)
    
    def transform_detections_to_pitch(self, detections: sv.Detections, 
                                    view_transformer: ViewTransformer) -> np.ndarray:
        """Transform detection bounding boxes to pitch coordinates.
        
        Args:
            detections: Detection results from YOLO
            view_transformer: ViewTransformer for frame-to-pitch conversion
            
        Returns:
            Array of transformed center points in pitch coordinates (N, 2)
        """
        if detections is None or len(detections.xyxy) == 0 or view_transformer is None:
            return np.array([]).reshape(0, 2)
        
        # Get center points of bounding boxes
        bboxes = detections.xyxy
        center_points = np.array([
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] 
            for bbox in bboxes
        ])
        
        # Transform to pitch coordinates
        pitch_points = self.homography_transformer.transform_points_to_pitch(
            center_points, view_transformer
        )
        
        return pitch_points if pitch_points is not None else np.array([]).reshape(0, 2)
    
    def create_tactical_frame(self, player_points: np.ndarray, ball_points: np.ndarray, 
                            referee_points: np.ndarray, frame_size: Tuple[int, int] = (1050, 680)) -> np.ndarray:
        """Create a tactical view frame showing positions on the pitch.
        
        Args:
            player_points: Player positions in pitch coordinates (N, 2)
            ball_points: Ball positions in pitch coordinates (N, 2) 
            referee_points: Referee positions in pitch coordinates (N, 2)
            frame_size: Size of output tactical frame (width, height)
            
        Returns:
            Tactical view frame as numpy array
        """
        # Create pitch visualization
        pitch_frame = draw_pitch(self.pitch_config)
        
        # Draw player positions
        if len(player_points) > 0:
            for point in player_points:
                if not np.isnan(point).any():
                    # Convert pitch coordinates to frame coordinates
                    x_ratio = point[0] / 12000  # Pitch width is 12000 units
                    y_ratio = point[1] / 7000   # Pitch height is 7000 units
                    
                    frame_x = int(x_ratio * frame_size[0])
                    frame_y = int(y_ratio * frame_size[1])
                    
                    # Draw player as circle
                    cv2.circle(pitch_frame, (frame_x, frame_y), 8, (0, 255, 0), -1)  # Green for players
        
        # Draw ball positions
        if len(ball_points) > 0:
            for point in ball_points:
                if not np.isnan(point).any():
                    x_ratio = point[0] / 12000
                    y_ratio = point[1] / 7000
                    
                    frame_x = int(x_ratio * frame_size[0])
                    frame_y = int(y_ratio * frame_size[1])
                    
                    # Draw ball as circle
                    cv2.circle(pitch_frame, (frame_x, frame_y), 6, (0, 0, 255), -1)  # Red for ball
        
        # Draw referee positions
        if len(referee_points) > 0:
            for point in referee_points:
                if not np.isnan(point).any():
                    x_ratio = point[0] / 12000
                    y_ratio = point[1] / 7000
                    
                    frame_x = int(x_ratio * frame_size[0])
                    frame_y = int(y_ratio * frame_size[1])
                    
                    # Draw referee as square
                    cv2.rectangle(pitch_frame, (frame_x-6, frame_y-6), 
                                (frame_x+6, frame_y+6), (255, 0, 0), -1)  # Blue for referees
        
        return pitch_frame
    
    def process_frame_for_tactical_analysis(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame for tactical analysis.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (tactical_frame, metadata_dict)
        """
        # Detect keypoints and objects
        keypoints = self.detect_frame_keypoints(frame)
        player_detections, ball_detections, referee_detections = self.detect_frame_objects(frame)
        
        # Get transformation matrix
        view_transformer = self.transform_keypoints_to_pitch(keypoints)
        
        # Transform detections to pitch coordinates
        player_pitch_points = self.transform_detections_to_pitch(player_detections, view_transformer)
        ball_pitch_points = self.transform_detections_to_pitch(ball_detections, view_transformer)
        referee_pitch_points = self.transform_detections_to_pitch(referee_detections, view_transformer)
        
        # Create tactical frame
        tactical_frame = self.create_tactical_frame(
            player_pitch_points, ball_pitch_points, referee_pitch_points
        )
        
        # Prepare metadata
        metadata = {
            'num_players': len(player_pitch_points),
            'num_balls': len(ball_pitch_points),
            'num_referees': len(referee_pitch_points),
            'transformation_valid': view_transformer is not None,
            'player_positions': player_pitch_points.tolist() if len(player_pitch_points) > 0 else [],
            'ball_positions': ball_pitch_points.tolist() if len(ball_pitch_points) > 0 else [],
            'referee_positions': referee_pitch_points.tolist() if len(referee_pitch_points) > 0 else []
        }
        
        return tactical_frame, metadata
    
    def analyze_video(self, video_path: str, output_path: str, frame_count: int = -1):
        """Analyze a complete video and create tactical analysis output.
        
        Args:
            video_path: Path to input video
            output_path: Path to save tactical analysis video
            frame_count: Number of frames to process (-1 for all frames)
        """
        if self.keypoint_model is None or self.detection_model is None:
            self.initialize_models()
        
        print("Reading video frames...")
        video_frames = read_video(video_path, frame_count=frame_count)
        
        print("Processing frames for tactical analysis...")
        tactical_frames = []
        all_metadata = []
        
        for i, frame in enumerate(video_frames):
            tactical_frame, metadata = self.process_frame_for_tactical_analysis(frame)
            tactical_frames.append(tactical_frame)
            all_metadata.append(metadata)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(video_frames)} frames")
        
        print("Writing tactical analysis video...")
        write_video(tactical_frames, output_path)
        
        print(f"Tactical analysis complete! Output saved to: {output_path}")
        return all_metadata
    
    def analyze_realtime(self, video_path: str, display_original: bool = True):
        """Run real-time tactical analysis on a video stream.
        
        Args:
            video_path: Path to input video or camera index (0 for webcam)
            display_original: Whether to display original video alongside tactical view
        """
        if self.keypoint_model is None or self.detection_model is None:
            self.initialize_models()

        print("Opening video stream...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        print("Starting real-time tactical analysis. Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Process frame for tactical analysis
            tactical_frame, metadata = self.process_frame_for_tactical_analysis(frame)
            
            # Display frames
            if display_original:
                # Resize frames for side-by-side display
                frame_height = 480
                frame_width = int(frame.shape[1] * (frame_height / frame.shape[0]))
                tactical_width = int(tactical_frame.shape[1] * (frame_height / tactical_frame.shape[0]))
                
                # Resize frames
                resized_frame = cv2.resize(frame, (frame_width, frame_height))
                resized_tactical = cv2.resize(tactical_frame, (tactical_width, frame_height))
                
                # Create combined display
                combined_frame = np.hstack([resized_frame, resized_tactical])
                
                # Add text overlay with metadata
                text_y = 30
                cv2.putText(combined_frame, f"Players: {metadata['num_players']}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(combined_frame, f"Ball: {metadata['num_balls']}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(combined_frame, f"Referees: {metadata['num_referees']}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(combined_frame, f"Transform: {'Valid' if metadata['transformation_valid'] else 'Invalid'}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (0, 255, 0) if metadata['transformation_valid'] else (0, 0, 255), 2)
                
                cv2.imshow("Soccer Tactical Analysis - Original | Tactical View", combined_frame)
            else:
                # Display only tactical frame
                cv2.imshow("Soccer Tactical Analysis", tactical_frame)
                
                # Add metadata as window title or overlay
                window_title = f"Players: {metadata['num_players']} | Ball: {metadata['num_balls']} | Referees: {metadata['num_referees']}"
                cv2.setWindowTitle("Soccer Tactical Analysis", window_title)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time tactical analysis stopped.")


if __name__ == "__main__":
    from keypoint_detection.keypoint_constants import keypoint_model_path
    from player_detection.detection_constants import model_path
    from constants import test_video
    
    # Example usage
    pipeline = TacticalPipeline(keypoint_model_path, model_path)
    
    # Choose analysis mode (uncomment desired option)
    
    # Option 1: Video analysis (saves to file)
    # output_path = test_video.replace('.mp4', '_tactical_analysis.mp4')
    # pipeline.analyze_video(test_video, output_path, frame_count=300)
    
    # Option 2: Real-time analysis (displays live)
    pipeline.analyze_realtime(test_video, display_original=True)
    
    # Option 3: Real-time analysis with webcam (use camera index 0)
    # pipeline.analyze_realtime(0, display_original=True)