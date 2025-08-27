import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import time
import supervision as sv
from tqdm import tqdm

from pipelines.detection_pipeline import DetectionPipeline
from pipelines.processing_pipeline import ProcessingPipeline
from player_tracking import TrackerManager
from player_clustering import ClusteringManager
from player_annotations import AnnotatorManager


class TrackingPipeline:
    """
    Complete tracking pipeline that combines detection, tracking, clustering and annotation.
    
    This pipeline provides a comprehensive solution for soccer video analysis by:
    - Object detection (players, ball, referees) using YOLO
    - Multi-object tracking using ByteTrack
    - Team assignment using SigLIP embeddings + UMAP + K-means clustering
    - Video annotation and output generation
    - Track extraction and ball interpolation support
    
    The pipeline is designed to work with the complete soccer analysis system
    and can be used standalone or integrated with other pipelines.
    """
    
    def __init__(self, model_path):
        """
        Initialize the tracking pipeline with all necessary models.
        
        Args:
            model_path: Path to the YOLO detection model
        """
        self.detection_pipeline = DetectionPipeline(model_path)
        self.processing_pipeline = ProcessingPipeline()
        self.tracker_manager = None
        self.clustering_manager = None
        self.annotator_manager = None
        
    def initialize_models(self):
        """Initialize all models required for the pipeline."""
        print("Initializing tracking pipeline models...")
        model_init_time = time.time()
        
        # Initialize detection pipeline
        print("Initializing detection pipeline...")
        self.detection_pipeline.initialize_model()
        
        # Initialize tracker
        print("Initializing tracker...")
        self.tracker_manager = TrackerManager()
        
        # Initialize clustering manager
        print("Initializing clustering manager...")
        self.clustering_manager = ClusteringManager()
        
        # Initialize annotator manager
        print("Initializing annotators...")
        self.annotator_manager = AnnotatorManager()
        
        model_init_time = time.time() - model_init_time
        print(f"Model initialization completed in {model_init_time:.2f}s")
        
    def collect_training_crops(self, video_path):
        """
        Collect player crops from video for training clustering models.
        
        Args:
            video_path: Path to training video
            
        Returns:
            List of player crop images
        """
        print("Collecting player crops for training...")
        
        # Get video frames
        frame_generator = sv.get_video_frames_generator(video_path, stride=12, end=120*24)
        
        # Extract player crops
        crops = []
        for frame in tqdm(frame_generator, desc='collecting_crops'):
            player_detections, _, _ = self.detection_pipeline.detect_frame_objects(frame)
            cropped_images = self.clustering_manager.embedding_extractor.get_player_crops(frame, player_detections)
            crops += cropped_images
        
        print(f"Collected {len(crops)} player crops")
        return crops
    
    def train_team_assignment_models(self, video_path):
        """
        Train UMAP and K-means models for team assignment.
        
        Args:
            video_path: Path to training video
            
        Returns:
            Trained clustering models
        """
        print("Training team assignment models...")
        training_time = time.time()
        
        # Collect training crops
        crops = self.collect_training_crops(video_path)
        
        # Train clustering models
        cluster_labels, reducer, cluster_model = self.clustering_manager.train_clustering_models(crops)
        
        training_time = time.time() - training_time
        print(f"Team assignment training completed in {training_time:.2f}s")
        
        return cluster_labels, reducer, cluster_model
    
    def detection_callback(self, frame):
        """
        Detection callback for processing individual frames.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of detection results (player, ball, referee)
        """
        detection_time = time.time()
        player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
        detection_time = time.time() - detection_time
        
        return player_detections, ball_detections, referee_detections, detection_time
    
    def tracking_callback(self, player_detections):
        """
        Tracking callback for updating player tracks.
        
        Args:
            player_detections: Player detection results
            
        Returns:
            Updated player detections with tracking information
        """
        return self.tracker_manager.process_tracking_for_frame(player_detections)
    
    def clustering_callback(self, frame, player_detections):
        """
        Clustering callback for team assignment.
        
        Args:
            frame: Input video frame
            player_detections: Player detection results
            
        Returns:
            Updated player detections with team assignments
        """
        assignment_time = time.time()
        cluster_labels = self.clustering_manager.get_cluster_labels(frame, player_detections)
        assignment_time = time.time() - assignment_time
        
        # Assign team labels
        player_detections.class_id = cluster_labels
        
        return player_detections, assignment_time

    def convert_detection_to_tracks(self, player_detections, ball_detections, referee_detections, tracks, index):
        """
        Convert detection results to track format for storage.
        
        Args:
            player_detections: Player detection results from YOLO
            ball_detections: Ball detection results from YOLO
            referee_detections: Referee detection results from YOLO
            tracks: Existing tracks dictionary to update
            index: Frame index for track storage
            
        Returns:
            Updated tracks dictionary with current frame detections
        """
        # Store player tracks with tracker IDs and class IDs
        if len(player_detections.xyxy) > 0:
            for tracker_id, bbox, class_id in zip(player_detections.tracker_id, player_detections.xyxy, player_detections.class_id):
                if index not in tracks['player']:
                    tracks['player'][index] = {}
                tracks['player'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
                
                # Store class IDs separately
                if index not in tracks['player_classids']:
                    tracks['player_classids'][index] = {}
                tracks['player_classids'][index][tracker_id] = class_id
        else:
            tracks['player'][index] = {-1: [None]*4}
            tracks['player_classids'][index] = {-1: None}
        
        # Store ball tracks (single detection per frame)
        if len(ball_detections.xyxy) > 0:
            for bbox in ball_detections.xyxy:
                tracks['ball'][index] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        else:
            tracks['ball'][index] = [None]*4
        
        # Store referee tracks with sequential IDs
        if len(referee_detections.xyxy) > 0:
            for tracker_id, bbox in zip(np.arange(len(referee_detections.xyxy)), referee_detections.xyxy):
                if index not in tracks['referee']:
                    tracks['referee'][index] = {}
                tracks['referee'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        else:
            tracks['referee'][index] = {-1: [None]*4}
    
        return tracks

    def get_tracks(self, frames):
        """
        Process video frames and extract tracks for all objects.
        
        This method processes each frame through detection and tracking pipelines,
        then converts the results to a structured track format for storage.
        
        Args:
            frames: List of video frames to process
            
        Returns:
            Dictionary containing tracks with structure:
            {
                'player': {frame_idx: {tracker_id: [x1, y1, x2, y2]}},
                'ball': {frame_idx: [x1, y1, x2, y2]},
                'referee': {frame_idx: {referee_id: [x1, y1, x2, y2]}}
            }
        """
        print("Processing frames and extracting tracks...")
        tracks = {
            'player': {},
            'ball': {},
            'referee': {},
            'player_classids': {},
        }
        
        for index, frame in tqdm(enumerate(frames), total=len(frames)):
            # Detection pipeline - detect objects in frame
            player_detections, ball_detections, referee_detections, det_time = self.detection_callback(frame)
            
            # Tracking pipeline - update player tracking with ByteTrack
            player_detections = self.tracking_callback(player_detections)
            
            # Team assignment (clustering callback)
            if player_detections is not None:
                player_detections, _ = self.clustering_callback(frame, player_detections)
            
            # Convert detections to structured track format
            tracks = self.convert_detection_to_tracks(player_detections, ball_detections, referee_detections, tracks, index)
        
        return tracks
    
    def annotate_frames(self, frames, tracks):
        """
        Annotate video frames with tracking and team assignment results.
        
        Args:
            frames: List of video frames
            tracks: Tracking results dictionary
            
        Returns:
            List of annotated frames
        """
        print("Annotating frames...")
        annotated_frames = []
        
        for index, frame in tqdm(enumerate(frames), total=len(frames)):
            # Get tracks for this frame
            player_tracks = tracks['player'][index]
            ball_tracks = tracks['ball'][index]
            referee_tracks = tracks['referee'][index]
            player_classids = tracks.get('player_classids', {}).get(index, None)
            
            # Clean up invalid tracks
            if -1 in player_tracks:
                player_tracks = None
                player_classids = None
            if -1 in referee_tracks:
                referee_tracks = None
            if (not all(ball_tracks)) or np.isnan(ball_tracks).all():
                ball_tracks = None
            
            # Convert to detections with stored class IDs
            player_detections, ball_detections, referee_detections = self.annotator_manager.convert_tracks_to_detections(
                player_tracks, ball_tracks, referee_tracks, player_classids
            )
            
            # Annotate frame
            annotated_frame = self.annotator_manager.annotate_all(
                frame, player_detections, ball_detections, referee_detections
            )
            annotated_frames.append(annotated_frame)
        
        return annotated_frames
    
    def track_in_video(self, video_path: str, output_path: str, frame_count: int = -1):
        """Analyze a complete video with tracking and team assignment.
        
        Args:
            video_path: Path to input video
            output_path: Path to save tracked video
            frame_count: Number of frames to process (-1 for all frames)
            train_models: Whether to train team assignment models from the video
        """
        self.initialize_models()
        
        # Train team assignment models
        print("Training team assignment models...")
        self.train_team_assignment_models(video_path)
        
        print("Reading video frames...")
        frames = self.processing_pipeline.read_video_frames(video_path, frame_count)
        
        print("Extracting tracks from video...")
        tracks = self.get_tracks(frames)
        
        print("Interpolating ball tracks...")
        tracks = self.processing_pipeline.interpolate_ball_tracks(tracks)
        
        print("Annotating frames with tracking results...")
        annotated_frames = self.annotate_frames(frames, tracks)
        
        print("Writing tracked video...")
        self.processing_pipeline.write_video_output(annotated_frames, output_path)
        
        print(f"Tracking complete! Output saved to: {output_path}")
        return tracks
    
    def track_realtime(self, video_path: str, display_metadata: bool = True):
        """Run real-time tracking analysis on a video stream.
        
        Args:
            video_path: Path to input video or camera index (0 for webcam)
            train_models: Whether to train team assignment models first
            display_metadata: Whether to display tracking metadata
        """
        import cv2
        
        self.initialize_models()
        
        # Train team assignment models
        print("Training team assignment models...")
        self.train_team_assignment_models(video_path)
        
        print("Opening video stream...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        print("Starting real-time tracking analysis. Press 'q' to quit.")
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Detection
            player_detections, ball_detections, referee_detections, det_time = self.detection_callback(frame)
            
            # Tracking
            player_detections = self.tracking_callback(player_detections)
            
            # Team assignment (if models are trained)
            if player_detections is not None:
                player_detections, _ = self.clustering_callback(frame, player_detections)
            
            # Annotate frame
            annotated_frame = self.annotator_manager.annotate_all(
                frame, player_detections, ball_detections, referee_detections
            )
            
            # Add metadata overlay if requested
            if display_metadata:
                text_y = 30
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Players: {len(player_detections.xyxy) if player_detections is not None else 0}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Ball: {len(ball_detections.xyxy) if ball_detections is not None else 0}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Referees: {len(referee_detections.xyxy) if referee_detections is not None else 0}", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30
                cv2.putText(annotated_frame, f"Detection Time: {det_time:.3f}s", 
                          (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Soccer Analysis - Real-time Tracking", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time tracking analysis stopped.")


if __name__ == "__main__":
    from player_detection.detection_constants import model_path, test_video
    
    # Example usage
    pipeline = TrackingPipeline(model_path)
    
    # Choose analysis mode (uncomment desired option)
    
    # Option 1: Video tracking analysis (saves to file)
    # output_path = test_video.replace('.mp4', '_tracked.mp4')
    # pipeline.track_in_video(test_video, output_path, frame_count=300, train_models=True)
    
    # Option 2: Real-time tracking analysis
    pipeline.track_realtime(test_video, display_metadata=True)