import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from keypoint_detection.keypoint_constants import KEYPOINT_NAMES


class HomographyTransformer:
    """
    Class for handling homography transformations between frame and pitch coordinates.
    """
    
    def __init__(self, confidence_threshold=0.3):
        """
        Initialize the transformer.
        
        Args:
            confidence_threshold: Minimum confidence to consider a keypoint valid
        """
        self.confidence_threshold = confidence_threshold
        self.CONFIG = SoccerPitchConfiguration()
        self.all_pitch_points = self._get_all_pitch_points()
        self.our_to_sports_mapping = self._get_keypoint_mapping()
    
    def _get_all_pitch_points(self):
        """Get all pitch reference points including extra points."""
        all_pitch_points = np.array(self.CONFIG.vertices)
        extra_pitch_points = np.array([
            [2932, 3500],  # Left Semicircle rightmost point
            [6000, 3500],  # Center Point
            [9069, 3500],  # Right semicircle rightmost point
        ])
        return np.concat((all_pitch_points, extra_pitch_points))
    
    def _get_keypoint_mapping(self):
        """Get mapping from our 29 keypoints to sports library's points."""
        return np.array([
            0,   # 0: sideline_top_left -> corner
            1,   # 1: big_rect_left_top_pt1 -> left penalty  
            9,   # 2: big_rect_left_top_pt2 -> left goal
            4,   # 3: big_rect_left_bottom_pt1 -> left goal
            12,  # 4: big_rect_left_bottom_pt2 -> left penalty
            2,   # 5: small_rect_left_top_pt1 -> left goal
            6,   # 6: small_rect_left_top_pt2 -> left goal
            3,   # 7: small_rect_left_bottom_pt1 -> left goal  
            7,   # 8: small_rect_left_bottom_pt2 -> left goal
            5,   # 9: sideline_bottom_left -> corner
            32,  # 10: left_semicircle_right -> penalty arc
            13,  # 11: center_line_top -> center
            16,  # 12: center_line_bottom -> center
            14,  # 13: center_circle_top -> center circle
            15,  # 14: center_circle_bottom -> center circle
            33,  # 15: field_center -> center circle
            24,  # 16: sideline_top_right -> corner
            25,  # 17: big_rect_right_top_pt1 -> right penalty
            17,  # 18: big_rect_right_top_pt2 -> right penalty
            28,  # 19: big_rect_right_bottom_pt1 -> right penalty
            20,  # 20: big_rect_right_bottom_pt2 -> right penalty
            26,  # 21: small_rect_right_top_pt1 -> right goal
            22,  # 22: small_rect_right_top_pt2 -> right goal
            27,  # 23: small_rect_right_bottom_pt1 -> right goal
            23,  # 24: small_rect_right_bottom_pt2 -> right goal
            29,  # 25: sideline_bottom_right -> corner
            34,  # 26: right_semicircle_left -> penalty arc
            30,  # 27: center_circle_left -> center_circle_left
            31,  # 28: center_circle_right -> center_circle_right
        ])
    
    def _filter_keypoints(self, detected_keypoints):
        """
        Filter keypoints based on confidence threshold.
        
        Args:
            detected_keypoints: Array of shape (1, 29, 3) with [x, y, confidence]
            
        Returns:
            Tuple of (frame_reference_points, pitch_reference_points, filter_mask)
        """
        if detected_keypoints.shape[0] == 0:
            return None, None, None
        
        keypoints = detected_keypoints[0]  # Take first detection
        
        # Create confidence filter
        filter_mask = keypoints[:, 2] > self.confidence_threshold
        
        if np.sum(filter_mask) < 4:
            print(f"Insufficient valid keypoints: {np.sum(filter_mask)} < 4")
            return None, None, None
        
        # Apply filter to get frame reference points
        frame_reference_points = keypoints[filter_mask, :2]  # Only x, y coordinates
        
        # Apply the same filter to get corresponding pitch points
        pitch_indices = self.our_to_sports_mapping[filter_mask]
        pitch_reference_points = self.all_pitch_points[pitch_indices]
        
        return frame_reference_points, pitch_reference_points, filter_mask
    
    def _create_view_transformer(self, source_points, target_points):
        """
        Create ViewTransformer from source to target points.
        
        Args:
            source_points: Source coordinate points
            target_points: Target coordinate points
            
        Returns:
            ViewTransformer object or None if creation fails
        """
        try:
            view_transformer = ViewTransformer(
                source=source_points,
                target=target_points
            )
            return view_transformer
        except ValueError as e:
            print(f"Error creating ViewTransformer: {e}")
            return None

    def transform_to_frame_keypoints(self, detected_keypoints):
        """
        Transform pitch keypoints to frame coordinates (for visualization).
        
        Args:
            detected_keypoints: Array of shape (1, 29, 3) with [x, y, confidence]
            
        Returns:
            Tuple of (transformed_points, view_transformer) or (None, None)
        """
        frame_ref_points, pitch_ref_points, filter_mask = self._filter_keypoints(detected_keypoints)
        
        if frame_ref_points is None:
            return None, None
        
        # Create ViewTransformer (source=pitch, target=frame)
        view_transformer = self._create_view_transformer(pitch_ref_points, frame_ref_points)
        
        if view_transformer is None:
            return None, None
        
        # Transform all pitch points to frame coordinates
        transformed_points = view_transformer.transform_points(points=self.all_pitch_points.copy())
        transformed_points = np.concat((transformed_points, np.ones((len(transformed_points), 1), dtype=np.float32)), axis=1)
        transformed_points = np.expand_dims(transformed_points, axis=0)
        
        return transformed_points, view_transformer
    
    def transform_to_pitch_keypoints(self, detected_keypoints):
        """
        Create ViewTransformer to transform frame coordinates to pitch coordinates.
        
        Args:
            detected_keypoints: Array of shape (1, 29, 3) with [x, y, confidence]
            
        Returns:
            ViewTransformer object or None if insufficient points
        """
        frame_ref_points, pitch_ref_points, filter_mask = self._filter_keypoints(detected_keypoints)
        
        if frame_ref_points is None:
            return None
        
        # Create ViewTransformer (source=frame, target=pitch)
        view_transformer = self._create_view_transformer(frame_ref_points, pitch_ref_points)
        
        return view_transformer
    
    def transform_points_to_pitch(self, points, view_transformer):
        """
        Transform points from frame coordinates to pitch coordinates.
        
        Args:
            points: Array of points in frame coordinates (N, 2)
            view_transformer: ViewTransformer object from transform_to_pitch_keypoints
            
        Returns:
            Transformed points in pitch coordinates or None if transformer is invalid
        """
        if view_transformer is None:
            return None
        
        try:
            transformed_points = view_transformer.transform_points(points=points)
            return transformed_points
        except Exception as e:
            print(f"Error transforming points: {e}")
            return None
    