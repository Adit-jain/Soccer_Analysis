"""Keypoint Detection Module for Soccer Analysis.

This module provides independent keypoint detection functionality for soccer field analysis,
following the modular architecture pattern of the Soccer Analysis project.
"""

from .detect_keypoints import (
    load_keypoint_model,
    detect_keypoints_in_frames,
    get_keypoint_detections,
    normalize_keypoints,
    denormalize_keypoints,
    filter_visible_keypoints,
    extract_field_corners,
    calculate_field_dimensions
)

from .keypoint_constants import (
    KEYPOINT_NAMES,
    KEYPOINT_CONNECTIONS,
    FIELD_CORNERS,
    NUM_KEYPOINTS,
    CONFIDENCE_THRESHOLD,
    FIFA_FIELD_LENGTH,
    FIFA_FIELD_WIDTH,
)

__all__ = [
    # Core detection functions
    'load_keypoint_model',
    'detect_keypoints_in_frames', 
    'get_keypoint_detections',
    'normalize_keypoints',
    'denormalize_keypoints',
    'filter_visible_keypoints',
    'extract_field_corners',
    'calculate_field_dimensions',
    
    # Constants
    'KEYPOINT_NAMES',
    'KEYPOINT_CONNECTIONS', 
    'FIELD_CORNERS',
    'NUM_KEYPOINTS',
    'CONFIDENCE_THRESHOLD',
    'FIFA_FIELD_LENGTH',
    'FIFA_FIELD_WIDTH',
    "KEYPOINT_COLOR", 
    "CONNECTION_COLOR", 
    "FIELD_CORNER_COLOR", 
    "TEXT_COLOR",
]