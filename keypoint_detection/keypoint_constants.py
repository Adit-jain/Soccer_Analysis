import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

# Keypoint model paths
keypoint_model_path = r"Models/Trained/yolov11_keypoints/First/weights/best.pt"
keypoint_model_path = PROJECT_DIR / keypoint_model_path

# Dataset configuration
calibration_dataset_path = r"F:\Datasets\SoccerNet\Data\calibration"
test_images_path = calibration_dataset_path + r"\images\test"
test_labels_path = calibration_dataset_path + r"\labels\test"
test_video = r"F:\Datasets\SoccerNet\Data\Samples\3_min_samp.mp4"
test_video_output = r"F:\Datasets\SoccerNet\Data\Samples\3_min_samp_keypoints.mp4"

# Keypoint model configuration
NUM_KEYPOINTS = 27
KEYPOINT_DIMENSIONS = 3  # (x, y, visibility)
CONFIDENCE_THRESHOLD = 0.3

# Soccer field keypoint names (27 keypoints)
KEYPOINT_NAMES = {
    0: "sideline_top_left",
    1: "big_rect_left_top_pt1", 
    2: "big_rect_left_top_pt2",
    3: "big_rect_left_bottom_pt1",
    4: "big_rect_left_bottom_pt2",
    5: "small_rect_left_top_pt1",
    6: "small_rect_left_top_pt2", 
    7: "small_rect_left_bottom_pt1",
    8: "small_rect_left_bottom_pt2",
    9: "sideline_bottom_left",
    10: "left_semicircle_right",
    11: "center_line_top",
    12: "center_line_bottom", 
    13: "center_circle_top",
    14: "center_circle_bottom",
    15: "field_center",
    16: "sideline_top_right",
    17: "big_rect_right_top_pt1",
    18: "big_rect_right_top_pt2",
    19: "big_rect_right_bottom_pt1",
    20: "big_rect_right_bottom_pt2",
    21: "small_rect_right_top_pt1",
    22: "small_rect_right_top_pt2",
    23: "small_rect_right_bottom_pt1", 
    24: "small_rect_right_bottom_pt2",
    25: "sideline_bottom_right",
    26: "right_semicircle_left"
}

# Keypoint connections for visualization
KEYPOINT_CONNECTIONS = [
    # Field boundary connections
    (0, 16),   # Top left to top right
    (0, 9),    # Top left to bottom left  
    (16, 25),  # Top right to bottom right
    (9, 25),   # Bottom left to bottom right
    
    # Left penalty area
    (1, 2),    # Big rect left top connections
    (3, 4),    # Big rect left bottom connections
    (1, 3),    # Big rect left vertical
    (2, 4),    # Big rect left vertical
    
    # Left goal area
    (5, 6),    # Small rect left top connections
    (7, 8),    # Small rect left bottom connections
    (5, 7),    # Small rect left vertical
    (6, 8),    # Small rect left vertical
    
    # Right penalty area
    (17, 18),  # Big rect right top connections
    (19, 20),  # Big rect right bottom connections
    (17, 19),  # Big rect right vertical
    (18, 20),  # Big rect right vertical
    
    # Right goal area
    (21, 22),  # Small rect right top connections
    (23, 24),  # Small rect right bottom connections
    (21, 23),  # Small rect right vertical
    (22, 24),  # Small rect right vertical
    
    # Center line and circle
    (11, 12),  # Center line vertical
    (13, 14),  # Center circle vertical
]

# Field corner keypoint indices
FIELD_CORNERS = {
    'top_left': 0,
    'top_right': 16, 
    'bottom_left': 9,
    'bottom_right': 25
}

# Standard FIFA field dimensions (in meters)
FIFA_FIELD_LENGTH = 105.0  # meters
FIFA_FIELD_WIDTH = 68.0    # meters

# Penalty area dimensions (in meters)
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.3

# Goal area dimensions (in meters)  
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.3

# Center circle radius (in meters)
CENTER_CIRCLE_RADIUS = 9.15

# Visualization colors (BGR format for OpenCV)
KEYPOINT_COLOR = (0, 255, 0)       # Green for keypoints
CONNECTION_COLOR = (255, 0, 0)     # Blue for connections
FIELD_CORNER_COLOR = (0, 0, 255)   # Red for field corners
TEXT_COLOR = (255, 255, 255)       # White for text