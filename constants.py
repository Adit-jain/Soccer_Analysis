"""
Global Configuration Constants for Soccer Analysis Project

This file contains the main configuration parameters for the soccer analysis system.
Update these paths according to your setup before running the system.

Model Download Instructions:
1. Visit: https://huggingface.co/Adit-jain/soccana
2. Download the trained model file
3. Place it in the path specified by model_path below
4. Update video paths to point to your test videos

For automatic model download, run:
    pip install huggingface_hub
    python -c "
    from huggingface_hub import hf_hub_download
    import os, shutil
    model_file = hf_hub_download(repo_id='Adit-jain/soccana', filename='best.pt')
    os.makedirs('Models/Trained/yolov11_sahi_1280/First/weights', exist_ok=True)
    shutil.copy(model_file, 'Models/Trained/yolov11_sahi_1280/First/weights/best.pt')
    print('Model downloaded successfully!')
    "
"""

import sys
from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Path to the trained YOLO model
# Download from: https://huggingface.co/Adit-jain/soccana
model_path = r"Models\Trained\yolov11_sahi_1280\First\weights\best.pt"
model_path = PROJECT_DIR / model_path

# Alternative model paths (uncomment if using different models)
# model_path = PROJECT_DIR / "Models/Pretrained/yolo11n.pt"  # Base YOLO model
# model_path = PROJECT_DIR / "path/to/your/custom/model.pt"   # Custom model

# =============================================================================
# VIDEO CONFIGURATION
# =============================================================================

# Input test video path
# UPDATE THIS: Point to your actual test video file
test_video = r"F:\Datasets\SoccerNet\Data\Samples\3_min_samp.mp4"

# Output video path
# UPDATE THIS: Where you want the tracked video to be saved
test_video_output = r"F:\Datasets\SoccerNet\Data\Samples\output2.mp4"

# Alternative video paths (examples)
# test_video = PROJECT_DIR / "test_videos/sample.mp4"
# test_video_output = PROJECT_DIR / "output/tracked_sample.mp4"

# For webcam input (use with real-time detection)
# webcam_index = 0  # Usually 0 for default webcam

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Team assignment training parameters
TRAINING_FRAME_STRIDE = 12        # Skip frames during training data collection
TRAINING_FRAME_LIMIT = 120 * 24   # Maximum frames for training (120*24 = ~2 mins at 24fps)

# Clustering parameters  
EMBEDDING_BATCH_SIZE = 24         # Batch size for SigLIP embedding extraction
UMAP_COMPONENTS = 3               # UMAP dimensionality reduction components
N_TEAMS = 2                       # Number of teams to cluster (usually 2)

# Tracking parameters
TRACKER_MATCH_THRESH = 0.5        # ByteTrack matching threshold
TRACKER_BUFFER_SIZE = 120         # Number of frames to keep in tracking buffer

# Ball interpolation
BALL_INTERPOLATION_LIMIT = 30     # Max frames to interpolate missing ball detections

# =============================================================================
# DETECTION CLASSES
# =============================================================================

CLASS_NAMES = {
    0: "Player",
    1: "Ball", 
    2: "Referee"
}

# Class colors for visualization (BGR format)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for players
    1: (0, 0, 255),    # Red for ball
    2: (255, 0, 0)     # Blue for referees
}

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# GPU settings
USE_GPU = True                    # Set to False to force CPU usage
GPU_DEVICE = 0                    # GPU device index (if multiple GPUs)

# Processing settings
MAX_VIDEO_FRAMES = -1             # Max frames to process (-1 for all frames)
OUTPUT_FPS = 30                   # Output video FPS

# Memory optimization
ENABLE_SAHI = False               # Enable SAHI for large image inference
SAHI_SLICE_HEIGHT = 640           # SAHI slice height
SAHI_SLICE_WIDTH = 640            # SAHI slice width
SAHI_OVERLAP_HEIGHT = 0.2         # SAHI overlap ratio
SAHI_OVERLAP_WIDTH = 0.2          # SAHI overlap ratio

# =============================================================================
# VALIDATION & DEBUGGING
# =============================================================================

# Validate paths on import
def validate_config():
    """Validate configuration and provide helpful messages."""
    issues = []
    
    # Check model path
    if not model_path.exists():
        issues.append(f"Model not found at: {model_path}")
        issues.append("Download from: https://huggingface.co/Adit-jain/soccana")
    
    # Check if video paths are still default (likely need updating)
    if "F:\\Datasets\\SoccerNet" in str(test_video):
        issues.append("test_video still uses default path - please update with your video file")
    
    if "F:\\Datasets\\SoccerNet" in str(test_video_output):
        issues.append("test_video_output still uses default path - please update")
    
    # Check video file exists (if not default)
    if "F:\\Datasets\\SoccerNet" not in str(test_video) and not Path(test_video).exists():
        issues.append(f"Input video not found: {test_video}")
    
    return issues

# Print configuration status
if __name__ == "__main__":
    print("=== Soccer Analysis Configuration ===")
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Model Path: {model_path}")
    print(f"Test Video: {test_video}")
    print(f"Output Path: {test_video_output}")
    print()
    
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues before running the system.")
    else:
        print("✅ Configuration looks good!")
        print("\nRun 'python main.py' to start the complete pipeline.")