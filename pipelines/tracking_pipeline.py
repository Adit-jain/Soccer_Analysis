import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import time
import supervision as sv
from tqdm import tqdm

from player_detection import load_detection_model, get_detections
from player_tracking import TrackerManager, process_tracking_for_frame
from player_clustering import ClusteringManager, train_clustering_models, get_cluster_labels
from player_annotations import AnnotatorManager


class TrackingPipeline:
    """
    Complete tracking pipeline that combines detection, tracking, clustering and annotation.
    """
    
    def __init__(self, model_path):
        """
        Initialize the tracking pipeline with all necessary models.
        
        Args:
            model_path: Path to the YOLO detection model
        """
        self.model_path = model_path
        self.detection_model = None
        self.tracker_manager = None
        self.clustering_manager = None
        self.annotator_manager = None
        
    def initialize_models(self):
        """Initialize all models required for the pipeline."""
        print("Initializing models...")
        model_init_time = time.time()
        
        # Load detection model
        print("Loading detection model...")
        self.detection_model = load_detection_model(self.model_path)
        
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
            player_detections, _, _ = get_detections(self.detection_model, frame)
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
        cluster_labels, reducer, cluster_model = train_clustering_models(
            crops, self.clustering_manager
        )
        
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
        player_detections, ball_detections, referee_detections = get_detections(
            self.detection_model, frame
        )
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
        tracker = self.tracker_manager.get_tracker()
        return process_tracking_for_frame(player_detections, tracker)
    
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
        cluster_labels = get_cluster_labels(frame, player_detections, self.clustering_manager)
        assignment_time = time.time() - assignment_time
        
        # Assign team labels
        player_detections.class_id = cluster_labels
        
        return player_detections, assignment_time
    
    def get_tracks(self, frames):
        """
        Process video frames and extract tracks for all objects.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary containing tracks for players, ball, and referees
        """
        print("Processing frames and extracting tracks...")
        tracks = {
            'player': {},
            'ball': {},
            'referee': {},
        }
        
        for index, frame in tqdm(enumerate(frames), total=len(frames)):
            # Detection
            player_detections, ball_detections, referee_detections, det_time = self.detection_callback(frame)
            
            # Tracking
            player_detections = self.tracking_callback(player_detections)
            
            # Store player tracks
            if len(player_detections.xyxy) > 0:
                for tracker_id, bbox in zip(player_detections.tracker_id, player_detections.xyxy):
                    if index not in tracks['player']:
                        tracks['player'][index] = {}
                    tracks['player'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
            else:
                tracks['player'][index] = {-1: [None]*4}
            
            # Store ball tracks
            if len(ball_detections.xyxy) > 0:
                for bbox in ball_detections.xyxy:
                    tracks['ball'][index] = [bbox[0], bbox[1], bbox[2], bbox[3]]
            else:
                tracks['ball'][index] = [None]*4
            
            # Store referee tracks
            if len(referee_detections.xyxy) > 0:
                for tracker_id, bbox in zip(np.arange(len(referee_detections.xyxy)), referee_detections.xyxy):
                    if index not in tracks['referee']:
                        tracks['referee'][index] = {}
                    tracks['referee'][index][tracker_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
            else:
                tracks['referee'][index] = {-1: [None]*4}
        
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
            
            # Clean up invalid tracks
            if -1 in player_tracks:
                player_tracks = None
            if -1 in referee_tracks:
                referee_tracks = None
            if (not all(ball_tracks)) or np.isnan(ball_tracks).all():
                ball_tracks = None
            
            # Convert to detections
            player_detections, ball_detections, referee_detections = self.annotator_manager.convert_tracks_to_detections(
                player_tracks, ball_tracks, referee_tracks
            )
            
            # Apply team assignment
            if player_detections is not None:
                player_detections, _ = self.clustering_callback(frame, player_detections)
            
            # Annotate frame
            annotated_frame = self.annotator_manager.annotate_all(
                frame, player_detections, ball_detections, referee_detections
            )
            annotated_frames.append(annotated_frame)
        
        return annotated_frames