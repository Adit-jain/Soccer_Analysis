import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv


class AnnotatorManager:
    """
    Manager class for annotation functionality.
    Handles initialization and configuration of various annotators.
    """
    
    def __init__(self):
        """Initialize all annotation tools."""
        self.ellipse_annotator = sv.EllipseAnnotator()
        self.triangle_annotator = sv.TriangleAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    
    def get_annotators(self):
        """Get all annotator instances as a tuple."""
        return (self.ellipse_annotator, self.triangle_annotator, self.label_annotator)


def annotate_players(frame: np.ndarray, player_detections, ball_detections, referee_detections,
                     ellipse_annotator: sv.EllipseAnnotator, 
                     triangle_annotator: sv.TriangleAnnotator, 
                     label_annotator: sv.LabelAnnotator) -> np.ndarray:
    """
    Annotate the players, ball, and referees on the frame.
    
    Args:
        frame: Input video frame
        player_detections: Player detection results
        ball_detections: Ball detection results  
        referee_detections: Referee detection results
        ellipse_annotator: Ellipse annotator for players/referees
        triangle_annotator: Triangle annotator for ball
        label_annotator: Label annotator for text
        
    Returns:
        Annotated frame with all detections visualized
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
            target_frame = triangle_annotator.annotate(target_frame, ball_detections)

    return target_frame


def convert_tracks_to_detections(player_tracks, ball_tracks, referee_tracks):
    """
    Convert tracking data back to supervision detections format.
    
    Args:
        player_tracks: Player tracking data for a frame
        ball_tracks: Ball tracking data for a frame
        referee_tracks: Referee tracking data for a frame
        
    Returns:
        Tuple of converted detection objects
    """
    # Get the player detections
    if player_tracks is not None:
        player_detections = sv.Detections(
            xyxy=np.array(list(player_tracks.values())),
            class_id=np.array([0] * len(player_tracks)),
            tracker_id=np.array(list(player_tracks.keys()))
        )
    else:
        player_detections = None

    # Get the ball detections
    if ball_tracks is not None:
        ball_detections = sv.Detections(
            xyxy=np.array([ball_tracks]),
            class_id=np.array([2]),
        )
    else:
        ball_detections = None

    # Get the referee detections
    if referee_tracks is not None:
        referee_detections = sv.Detections(
            xyxy=np.array(list(referee_tracks.values())),
            class_id=np.array([3] * len(referee_tracks)),
            tracker_id=np.array(list(referee_tracks.keys()))
        )
    else:
        referee_detections = None

    return player_detections, ball_detections, referee_detections