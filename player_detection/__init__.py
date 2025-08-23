import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_detection.detect_players import load_detection_model, get_detections, detect_objects_in_frames