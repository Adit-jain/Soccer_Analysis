import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv
import cv2


class AnnotatorManager:
    """
    Manager class for all annotation functionality.
    Provides a unified interface for annotating players, ball, referees, and keypoints.
    """

    def __init__(self, edges=None):
        """Initialize all annotation tools."""
        # Basic annotators
        self.ellipse_annotator = sv.EllipseAnnotator()
        self.triangle_annotator = sv.TriangleAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()

        # Keypoint annotators
        self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=8)
        self.edge_annotator = sv.EdgeAnnotator(color=sv.Color.BLUE, thickness=3, edges=edges)

    def annotate_players(self, frame: np.ndarray, player_detections: sv.Detections) -> np.ndarray:
        """
        Annotate only players on the frame.

        Args:
            frame: Input video frame
            player_detections: Player detection results

        Returns:
            Annotated frame with player detections visualized
        """
        if player_detections is None or len(player_detections.xyxy) == 0:
            return frame

        if player_detections.tracker_id is None:
            player_detections.tracker_id = np.arange(len(player_detections.xyxy))
        player_labels = [f'#{tracker_id}' for tracker_id in player_detections.tracker_id]
        frame = self.ellipse_annotator.annotate(frame, player_detections)
        frame = self.label_annotator.annotate(frame, detections=player_detections, labels=player_labels)

        return frame

    def annotate_ball(self, frame: np.ndarray, ball_detections: sv.Detections) -> np.ndarray:
        """
        Annotate ball detections on frame.

        Args:
            frame: Input video frame
            ball_detections: Ball detection results

        Returns:
            Annotated frame with ball detections
        """
        if ball_detections is not None and len(ball_detections.xyxy) > 0:
            return self.triangle_annotator.annotate(frame, ball_detections)
        return frame

    def annotate_referees(self, frame: np.ndarray, referee_detections: sv.Detections) -> np.ndarray:
        """
        Annotate referee detections on frame.

        Args:
            frame: Input video frame
            referee_detections: Referee detection results

        Returns:
            Annotated frame with referee detections
        """
        if referee_detections is not None and len(referee_detections.xyxy) > 0:
            if referee_detections.tracker_id is None:
                referee_detections.tracker_id = np.arange(len(referee_detections.xyxy))
            return self.ellipse_annotator.annotate(frame, referee_detections)
        return frame

    def annotate_all(self, frame: np.ndarray, player_detections, ball_detections, referee_detections) -> np.ndarray:
        """
        Annotate players, ball, and referees on the frame using separate methods.

        Args:
            frame: Input video frame
            player_detections: Player detection results
            ball_detections: Ball detection results
            referee_detections: Referee detection results

        Returns:
            Annotated frame with all detections visualized
        """
        target_frame = frame.copy()

        # Annotate each type separately
        target_frame = self.annotate_players(target_frame, player_detections)
        target_frame = self.annotate_ball(target_frame, ball_detections)
        target_frame = self.annotate_referees(target_frame, referee_detections)

        return target_frame

    def annotate_bboxes(self, frame: np.ndarray, detections: sv.Detections, class_names: dict = None) -> np.ndarray:
        """
        Annotate frame with object detections bboxes.

        Args:
            frame: Input frame
            detections: Supervision Detections object
            class_names: Dictionary mapping class IDs to names

        Returns:
            Annotated frame with detection boxes and labels
        """
        if detections is None or len(detections.xyxy) == 0:
            return frame

        annotated_frame = frame.copy()

        # Annotate with boxes
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)

        # Add labels if class_names provided
        if class_names is not None:
            labels = []
            for class_id, conf in zip(detections.class_id, detections.confidence):
                class_name = class_names.get(class_id, f'Class {class_id}')
                labels.append(f"{class_name} {conf:.2f}")
            annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame

    def annotate_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.3,
                          draw_vertices: bool = True, draw_edges = None, draw_labels: bool = True,
                          KEYPOINT_CONNECTIONS=[], KEYPOINT_NAMES={}, 
                          KEYPOINT_COLOR=(0, 255, 0), CONNECTION_COLOR=(255, 0, 0), TEXT_COLOR=(255, 255, 255)) -> np.ndarray:
        """
        Annotate frame with detected keypoints using Vertex and Edge annotators.

        Args:
            frame: Input frame to annotate
            keypoints: Detected keypoints array with shape (N, 27, 3)
            confidence_threshold: Minimum confidence to draw keypoint
            draw_vertices: Whether to draw keypoint vertices
            draw_edges: Whether to draw connections between keypoints
            draw_labels: Whether to draw keypoint labels

        Returns:
            Annotated frame
        """

        # Draw keypoints and connections for each detection
        for kpts in keypoints:

            # Draw keypoint connections
            if draw_edges:
                for connection in KEYPOINT_CONNECTIONS:
                    pt1_idx, pt2_idx = connection
                    pt1 = kpts[pt1_idx]
                    pt2 = kpts[pt2_idx]

                    # Only draw if both points are visible
                    if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                        cv2.line(frame,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            CONNECTION_COLOR, 2)

            # Draw keypoints
            for kpt_idx, kpt in enumerate(kpts):
                if kpt[2] > confidence_threshold:  # Check visibility
                    x, y = int(kpt[0]), int(kpt[1])

                    # Draw keypoint circle
                    cv2.circle(frame, (x, y), 5, KEYPOINT_COLOR, -1)

                    # Draw keypoint label
                    if draw_labels:
                        label = f"{kpt_idx}: {KEYPOINT_NAMES.get(kpt_idx, 'Unknown')}"
                        cv2.putText(frame, label, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

        return frame

    def convert_tracks_to_detections(self, player_tracks, ball_tracks, referee_tracks):
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