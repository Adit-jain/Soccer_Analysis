import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

model_path = r"Models/Trained/yolov11_sahi_1280/First/weights/best.pt"
model_path = PROJECT_DIR / model_path

test_video = r"D:\Datasets\SoccerNet\Data\Samples\1_720p.mkv"
test_video_output = r"D:\Datasets\SoccerNet\Data\Samples\1_720p_output.mkv"
test_image_dir = r"D:\Datasets\SoccerNet\Data\tracking\images\challenge\SNMOT-021\img1"