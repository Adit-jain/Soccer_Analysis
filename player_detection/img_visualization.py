from ultralytics import YOLO
import os
import random
from constants import test_image_dir, model_path

model_1 = YOLO(model_path)
# model_2 = YOLO(model_path) # Uncomment this line to use the second model

# Load Sample number of images
samples = 10
img_paths = [os.path.join(test_image_dir, img) for img in os.listdir(test_image_dir)]
img_paths = random.sample(img_paths, samples)

# Predict the images
result = []
for model in [model_1]:
    result.append(model(img_paths))

# Show the images for each model
for i in range(samples):
    for model in result:
        model[i].show()
    input("Press Enter to continue...")



