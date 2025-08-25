import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

model_path = r"Models/Trained/yolov11_sahi_1280/Model/weights/best.pt"
model_path = PROJECT_DIR / model_path

test_video = r"F:\Datasets\SoccerNet\Data\Samples\3_min_samp.mp4"
test_video_output = r"F:\Datasets\SoccerNet\Data\Samples\3_min_samp_dect.mp4"
test_image_dir = r"D:\Datasets\SoccerNet\Data\tracking\images\challenge\SNMOT-021\img1"