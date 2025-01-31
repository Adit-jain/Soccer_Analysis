"""File to Fine-Tune a YOLO model"""

from ultralytics import YOLO
from train_args import (dataset_name, epochs, img_size, batch_size, model_name, run, start_model_name, start_model_run,
                  resume, save_period, single_cls, freeze, lr0, lrf, dropout)
import os


if __name__ == "__main__":

    # Get current directory path
    cur_dir_path = os.path.dirname(__file__)

    # Train on Original YOLO model or new model
    if 'original_' in start_model_name:
        model_path = os.path.abspath(os.path.join(cur_dir_path, fr"../Models/Pretrained/{start_model_name.split('original_')[-1]}.pt"))
    else:
        model_path = os.path.abspath(os.path.join(cur_dir_path, fr"../Models/Trained/{start_model_name}/{start_model_run}/weights/best.pt"))
    print(f"Model : {model_path}")
    model = YOLO(model_path)

    # Get YAML path
    yaml_path_txt = os.path.abspath(os.path.join(cur_dir_path, fr"../Data_utils/{dataset_name}/yaml_path.txt"))
    with open(yaml_path_txt, 'r') as f:
        yaml_path = f.readline()
    print(f"YAML path : {yaml_path}")

    # Training
    results = model.train(data=fr"{yaml_path}",
                          epochs=epochs,
                          imgsz=img_size,
                          batch=batch_size,
                          project=os.path.abspath(os.path.join(cur_dir_path, fr'../Models/Trained/{model_name}')),
                          name=f'{run}',
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