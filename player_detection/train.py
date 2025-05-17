"""File to Fine-Tune a YOLO model"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from ultralytics import YOLO
from player_detection.train_args import (dataset_yaml_path, model_path, epochs, img_size, batch_size, model_name, run, resume, save_period, single_cls, freeze, lr0, lrf, dropout)
import os

if __name__ == "__main__":
    # Model path
    print(f"Model : {model_path}")
    model = YOLO(model_path)

    # YAML path
    print(f"YAML path : {dataset_yaml_path}")

    # Training
    results = model.train(data=fr"{dataset_yaml_path}",
                          epochs=epochs,
                          imgsz=img_size,
                          batch=batch_size,
                          project= PROJECT_DIR / fr'Models/Trained/{model_name}',
                          name=run,
                          seed=44,
                          resume=resume,
                          save=True,
                          save_period=save_period,
                          single_cls=single_cls,
                          freeze=freeze,
                          lr0=lr0,
                          lrf=lrf,
                          dropout=dropout,
                          plots=True)