import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_detection.detect_players import load_detection_model, detect_players_in_frames, get_detections, detect_players_images, detect_players_video, detect_players_realtime