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
    target_frame = frame.copy()

    # Player annotations
    if player_detections is not None:
        if len(player_detections.xyxy) > 0:
            if player_detections.tracker_id is None:
                player_detections.tracker_id = np.arange(len(player_detections.xyxy))
            player_labels = [f'#{tracker_id}' for tracker_id in player_detections.tracker_id]
            target_frame = ellipse_annotator.annotate(target_frame, player_detections)
            target_frame = label_annotator.annotate(target_frame, detections=player_detections, labels=player_labels)

    # Referee annotations
    if referee_detections is not None:
        if len(referee_detections.xyxy) > 0:
            if referee_detections.tracker_id is None:
                referee_detections.tracker_id = np.arange(len(referee_detections.xyxy))
            target_frame = ellipse_annotator.annotate(target_frame, referee_detections)

    # Ball annotations
    if ball_detections is not None:
        if len(ball_detections.xyxy) > 0:
            target_frame = traingle_annotator.annotate(target_frame, ball_detections)

    # Return the annotated frame
    return target_frame