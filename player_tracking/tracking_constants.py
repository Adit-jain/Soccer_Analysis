import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

model_path = r"C:\Studies\Artificial Intelligence\ObjectDetection\Soccer_analysis\Models\Trained\yolov11_1280\First\weights\best.pt"
model_path = os.path.join(PROJECT_PATH, model_path)

test_video = r"D:\Datasets\SoccerNet\Data\Samples\3_min_samp.mp4"
test_video_output = r"D:\Datasets\SoccerNet\Data\Samples\output.mp4"