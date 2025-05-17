import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv

def init_tracking_models():
    """
    Initialize the tracking models for the player tracking system
    """
    tracker = sv.ByteTrack()
    tracker.match_thresh = 0.5
    tracker.track_buffer = 120
    ellipse_annotator = sv.EllipseAnnotator()
    traingle_annotator = sv.TriangleAnnotator()
    label_annotator = sv.LabelAnnotator()

    return tracker, ellipse_annotator, traingle_annotator, label_annotator

def update_player_detections(player_detections, tracker: sv.ByteTrack):
    """
    Update the player detections with the tracker
    """
    player_detections = tracker.update_with_detections(player_detections)
    return player_detections

def annotate_players(frame: np.ndarray, player_detections, ball_detections, referee_detections,
                     ellipse_annotator: sv.EllipseAnnotator, 
                     traingle_annotator: sv.TriangleAnnotator, 
                     label_annotator: sv.LabelAnnotator) -> np.ndarray:
    """
    Annotate the players and ball on the frame
    """
    # Get the labels
    player_labels = [f'#{tracker_id}' for tracker_id in player_detections.tracker_id]

    # Annotate the frame
    player_annotated_frame = ellipse_annotator.annotate(frame.copy(), player_detections)
    referee_annotated_frame = ellipse_annotator.annotate(player_annotated_frame, referee_detections)
    ball_annotated_frame = traingle_annotator.annotate(referee_annotated_frame, ball_detections)
    labeled_frame = label_annotator.annotate(ball_annotated_frame, detections=player_detections, labels=player_labels)

    return labeled_frame