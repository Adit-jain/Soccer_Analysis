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


def transform_to_pitch_keypoints(detected_keypoints, confidence_threshold=0.3):
    """
    Create ViewTransformer using the same filtering approach as the sports library example.
    
    Args:
        detected_keypoints: Array of shape (1, 27, 3) with [x, y, confidence] 
        confidence_threshold: Minimum confidence to consider a keypoint valid
        
    Returns:
        ViewTransformer object or None if insufficient points
    """
    if detected_keypoints.shape[0] == 0:
        return None
    
    keypoints = detected_keypoints[0]  # Take first detection
    
    # Create confidence filter - same as the example code
    filter_mask = keypoints[:, 2] > confidence_threshold
    
    if np.sum(filter_mask) < 4:
        print(f"Insufficient valid keypoints: {np.sum(filter_mask)} < 4")
        return None, None
    
    # Apply filter to get frame reference points (detected keypoints)
    frame_reference_points = keypoints[filter_mask, :2]  # Only x, y coordinates
    
    # Map our 27 keypoints to the sports library's 32 points based on correct field positions
    # Sports field: (0,0) = left-top corner, (12000,7000) = right-bottom corner
    our_to_sports_mapping = np.array([
        0,   # 0: sideline_top_left -> corner
        1,   # 1: big_rect_left_top_pt1 -> left penalty  
        9,   # 2: big_rect_left_top_pt2 -> left goal
        4,   # 3: big_rect_left_bottom_pt1 -> left goal
        12,   # 4: big_rect_left_bottom_pt2 -> left penalty
        2,   # 5: small_rect_left_top_pt1 -> left goal
        6,   # 6: small_rect_left_top_pt2 -> left goal
        3,   # 7: small_rect_left_bottom_pt1 -> left goal  
        7,   # 8: small_rect_left_bottom_pt2 -> left goal
        5,  # 9: sideline_bottom_left -> corner
        32,   # 10: left_semicircle_right -> penalty arc
        13,  # 11: center_line_top -> center
        16,  # 12: center_line_bottom -> center
        14,  # 13: center_circle_top -> center circle
        15,  # 14: center_circle_bottom -> center circle
        33,  # 15: field_center -> center circle
        24,   # 16: sideline_top_right -> corner
        25,  # 17: big_rect_right_top_pt1 -> right penalty
        17,  # 18: big_rect_right_top_pt2 -> right penalty
        28,  # 19: big_rect_right_bottom_pt1 -> right penalty
        20,  # 20: big_rect_right_bottom_pt2 -> right penalty
        26,  # 21: small_rect_right_top_pt1 -> right goal
        22,  # 22: small_rect_right_top_pt2 -> right goal
        27,  # 23: small_rect_right_bottom_pt1 -> right goal
        23,  # 24: small_rect_right_bottom_pt2 -> right goal
        29,  # 25: sideline_bottom_right -> corner
        34   # 26: right_semicircle_left -> penalty arc
    ])
    
    # Get sports library pitch points
    CONFIG = SoccerPitchConfiguration()
    all_pitch_points = np.array(CONFIG.vertices)
    extra_pitch_points = np.array([
    [2932, 3500], # Left Semicircle rightmost point
    [6000, 3500], # Center Point
    [9069, 3500], # Right semicircle rightmost point
    ])
    all_pitch_points = np.concat((all_pitch_points, extra_pitch_points))
    
    # Apply the same filter to get corresponding pitch points
    pitch_indices = our_to_sports_mapping[filter_mask]
    pitch_reference_points = all_pitch_points[pitch_indices]
    
    # Create ViewTransformer (source=pitch, target=image) 
    try:
        view_transformer = ViewTransformer(
            source=pitch_reference_points,
            target=frame_reference_points
        )
    except ValueError as e:
        print(f"Error creating ViewTransformer: {e}")
        return None, None
    
    # Transform all points
    transformed_points = view_transformer.transform_points(points=all_pitch_points.copy())
    transformed_points = np.concat((transformed_points, np.ones((len(transformed_points), 1), dtype=np.float32)), axis=1)
    transformed_points = np.expand_dims(transformed_points, axis=0)

    return transformed_points, view_transformer
    