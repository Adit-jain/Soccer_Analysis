import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv


class TrackerManager:
    """
    Manager class for player tracking functionality.
    Handles initialization and configuration of tracking models.
    """
    
    def __init__(self, match_thresh=0.5, track_buffer=120):
        """
        Initialize the tracker with configurable parameters.
        
        Args:
            match_thresh (float): Matching threshold for tracking
            track_buffer (int): Number of frames to keep tracking buffer
        """
        self.tracker = sv.ByteTrack()
        self.tracker.match_thresh = match_thresh
        self.tracker.track_buffer = track_buffer
    
    def get_tracker(self):
        """Get the configured tracker instance."""
        return self.tracker
    
    def update_player_detections(self, player_detections):
        """
        Update the player detections with the tracker.
        
        Args:
            player_detections: Detection results from YOLO
            
        Returns:
            Updated detections with tracker IDs
        """
        player_detections = self.tracker.update_with_detections(player_detections)
        return player_detections
    
    def process_tracking_for_frame(self, player_detections):
        """
        Process tracking for a single frame.
        
        Args:
            player_detections: Player detections for the frame
            
        Returns:
            Updated player detections with tracking information
        """
        if len(player_detections.xyxy) > 0:
            player_detections = self.update_player_detections(player_detections)
        return player_detections