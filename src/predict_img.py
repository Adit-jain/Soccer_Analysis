from ultralytics import YOLO
import os
import random

img_dir = r"D:\Datasets\SoccerNet\Data\tracking\images\val\SNMOT-127\img1"

model_3 = YOLO(r"Models/Trained/yolov11/Third/weights/best.pt")
model_2 = YOLO(r"Models/Trained/yolov11/Second/weights/best.pt")

img_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
img_paths = random.sample(img_paths, 10)

result = []
for model in [model_3, model_3]:
    result.append(model(img_paths))

for i in range(10):
    for model in result:
        model[i].show()
    input("Press Enter to continue...")



