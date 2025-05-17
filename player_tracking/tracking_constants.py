import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

model_path = r"Models\Trained\yolov11_1280\First\weights\best.pt"
model_path = PROJECT_DIR / model_path

test_video = r"D:\Datasets\SoccerNet\Data\Samples\3_min_samp.mp4"
test_video_output = r"D:\Datasets\SoccerNet\Data\Samples\output.mp4"