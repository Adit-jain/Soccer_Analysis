import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_tracking.tracking import TrackerManager

__all__ = ["TrackerManager"]