import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

dataset_yaml_path = r"D:\Datasets\SoccerAnalysis_Final\V1\data.yaml"
epochs = 30
img_size = 1280
batch_size = 8
model_name = 'yolov11_sahi_1280'
run = 'First'
start_model_name = 'original_yolo11n'
start_model_run = 'First'
resume = False
save_period = 1
single_cls = False
freeze = False
lr0 = 0.01
lrf = 0.01
dropout = 0.3

if 'original_' in start_model_name:
    model_path = fr"Models/Pretrained/{start_model_name.split('original_')[-1]}.pt"
else:
    model_path = fr"Models/Trained/{start_model_name}/{start_model_run}/weights/best.pt"

model_path = PROJECT_DIR / model_path