import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from ultralytics import YOLO
from player_detection.train_args import dataset_yaml_path, model_path, model_name, run, start_model_name, start_model_run

if __name__ == "__main__":

    # Model path
    print(f"Model : {model_path}")
    model = YOLO(model_path)

    # YAML path
    print(f"YAML path : {dataset_yaml_path}")

    # Validation
    metrics = model.val(data = fr"{dataset_yaml_path}",
                        project = PROJECT_DIR / fr'Models/Trained/{model_name}',
                        name = f'{run}_val')