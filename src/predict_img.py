from ultralytics import YOLO
import os
import random

img_dir = r"D:\Datasets\SoccerNet\Data\tracking\images\val\SNMOT-127\img1"

model_11 = YOLO(r"../Models/Pretrained/yolo11l.pt")
# model_10 = YOLO(r"yolov10l.pt")
# model_9 = YOLO(r"yolov9c.pt")
model_8 = YOLO(r"../pretrained_models/yolov8l.pt")

img_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
img_paths = random.sample(img_paths, 10)

result = []
for model in [model_8, model_11]:
    result.append(model(img_paths))

for i in range(10):
    for model in result:
        model[i].show()
    input("Press Enter to continue...")



