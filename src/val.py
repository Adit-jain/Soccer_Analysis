from ultralytics import YOLO
from args import dataset_name, model_name, run, start_model_name, start_model_run

if __name__ == "__main__":
    # Train on Original YOLO model or new model
    if 'original_' in start_model_name:
        model_path = fr"../Models/Pretrained/{start_model_name.split('original_')[-1]}.pt"
    else:
        model_path = fr"../Models/Trained/{start_model_name}/{start_model_run}/weights/best.pt"
    print(f"Model : {model_path}")
    model = YOLO(model_path)

    # Get YAML path
    yaml_path_txt = fr"../Data_utils/{dataset_name}/yaml_path.txt"
    with open(yaml_path_txt, 'r') as f:
        yaml_path = f.readline()
    print(f"YAML path : {yaml_path}")

    # Validation
    metrics = model.val(data=fr"{yaml_path}",
                        project=fr'../Models/Trained/{model_name}',
                        name=f'{run}_val')