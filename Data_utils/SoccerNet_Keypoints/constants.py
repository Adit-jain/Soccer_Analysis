import os
from pathlib import Path

dataset_dir = r"F:\Datasets\SoccerNet\Data"
calibration_dir = Path(dataset_dir) / 'calibration'
soccernet_password = "<password>"

# Keypoint Detection Constants

# Standard football field dimensions (in meters)
FIELD_LENGTH = 105.0  # FIFA standard: 100-110m
FIELD_WIDTH = 68.0    # FIFA standard: 64-75m

# Goal dimensions
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44

# Penalty areas
BIG_BOX_LENGTH = 16.5
BIG_BOX_WIDTH = 40.32

SMALL_BOX_LENGTH = 5.5
SMALL_BOX_WIDTH = 18.32

# Center circle
CENTER_CIRCLE_RADIUS = 9.15

# Standard field keypoints in real-world coordinates (meters)
# Origin at center of the field
STANDARD_FIELD_KEYPOINTS = {
    # Center points
    'center_point': (0.0, 0.0),
    'center_line_top': (0.0, FIELD_WIDTH/2),
    'center_line_bottom': (0.0, -FIELD_WIDTH/2),
    
    # Left goal area (attacking left)
    'left_goal_line_top': (-FIELD_LENGTH/2, GOAL_WIDTH/2),
    'left_goal_line_bottom': (-FIELD_LENGTH/2, -GOAL_WIDTH/2),
    'left_small_box_top_left': (-FIELD_LENGTH/2, SMALL_BOX_WIDTH/2),
    'left_small_box_top_right': (-FIELD_LENGTH/2 + SMALL_BOX_LENGTH, SMALL_BOX_WIDTH/2),
    'left_small_box_bottom_left': (-FIELD_LENGTH/2, -SMALL_BOX_WIDTH/2),
    'left_small_box_bottom_right': (-FIELD_LENGTH/2 + SMALL_BOX_LENGTH, -SMALL_BOX_WIDTH/2),
    'left_big_box_top_left': (-FIELD_LENGTH/2, BIG_BOX_WIDTH/2),
    'left_big_box_top_right': (-FIELD_LENGTH/2 + BIG_BOX_LENGTH, BIG_BOX_WIDTH/2),
    'left_big_box_bottom_left': (-FIELD_LENGTH/2, -BIG_BOX_WIDTH/2),
    'left_big_box_bottom_right': (-FIELD_LENGTH/2 + BIG_BOX_LENGTH, -BIG_BOX_WIDTH/2),
    
    # Right goal area (attacking right)
    'right_goal_line_top': (FIELD_LENGTH/2, GOAL_WIDTH/2),
    'right_goal_line_bottom': (FIELD_LENGTH/2, -GOAL_WIDTH/2),
    'right_small_box_top_left': (FIELD_LENGTH/2 - SMALL_BOX_LENGTH, SMALL_BOX_WIDTH/2),
    'right_small_box_top_right': (FIELD_LENGTH/2, SMALL_BOX_WIDTH/2),
    'right_small_box_bottom_left': (FIELD_LENGTH/2 - SMALL_BOX_LENGTH, -SMALL_BOX_WIDTH/2),
    'right_small_box_bottom_right': (FIELD_LENGTH/2, -SMALL_BOX_WIDTH/2),
    'right_big_box_top_left': (FIELD_LENGTH/2 - BIG_BOX_LENGTH, BIG_BOX_WIDTH/2),
    'right_big_box_top_right': (FIELD_LENGTH/2, BIG_BOX_WIDTH/2),
    'right_big_box_bottom_left': (FIELD_LENGTH/2 - BIG_BOX_LENGTH, -BIG_BOX_WIDTH/2),
    'right_big_box_bottom_right': (FIELD_LENGTH/2, -BIG_BOX_WIDTH/2),
    
    # Corner points
    'top_left_corner': (-FIELD_LENGTH/2, FIELD_WIDTH/2),
    'top_right_corner': (FIELD_LENGTH/2, FIELD_WIDTH/2),
    'bottom_left_corner': (-FIELD_LENGTH/2, -FIELD_WIDTH/2),
    'bottom_right_corner': (FIELD_LENGTH/2, -FIELD_WIDTH/2),
}

# Mapping from detected keypoints to standard field keypoints
KEYPOINT_MAPPING = {
    'big_box_top_corner': 'left_big_box_top_right',
    'big_box_bottom_corner': 'left_big_box_bottom_right',
    'small_box_top_corner': 'left_small_box_top_right',
    'small_box_bottom_corner': 'left_small_box_bottom_right',
    'goal_left_top_corner': 'left_goal_line_top',
    'goal_right_top_corner': 'left_goal_line_top',  # This might need adjustment
    'sideline_big_box_top': 'left_big_box_top_left',
    'sideline_big_box_main': 'left_big_box_bottom_left'
}