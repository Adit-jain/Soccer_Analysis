import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

model_path = r"Models/Trained/yolov11_1280/First/weights/best.pt"
model_path = os.path.join(PROJECT_PATH, model_path)

test_video = r"D:\Datasets\SoccerNet\Data\Samples\1_720p.mkv"
test_video_output = r"D:\Datasets\SoccerNet\Data\Samples\1_720p_output.mkv"

test_image_dir = r"D:\Datasets\SoccerNet\Data\tracking\images\challenge\SNMOT-021\img1"