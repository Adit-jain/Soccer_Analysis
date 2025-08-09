import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
print(PROJECT_DIR)
sys.path.append(str(PROJECT_DIR))

from pathlib import Path
import os, tqdm
import json
from Data_utils.SoccerNet_Keypoints.line_intersections import LineIntersectionCalculator
from Data_utils.SoccerNet_Keypoints.constants import calibration_dir

# Create output directories
output_json_dir = calibration_dir / 'keypoints'
output_image_dir = calibration_dir / 'keypoints_images'
output_json_dir.mkdir(parents=True, exist_ok=True)
output_image_dir.mkdir(parents=True, exist_ok=True)

# Initialize Line Intersection Calculator
calculator = LineIntersectionCalculator()

# Iterate through each dataset type
for dataset_type in os.listdir(calibration_dir):
    if dataset_type in ['train', 'test', 'valid']:
        print(f"Processing {dataset_type} dataset...")
        dataset_path = calibration_dir / dataset_type

        # Create keypoint directory and image directory for each dataset type
        keypoint_dir_path = output_json_dir / dataset_type
        keypoint_dir_path.mkdir(parents=True, exist_ok=True)
        image_dir_path = output_image_dir / dataset_type
        image_dir_path.mkdir(parents=True, exist_ok=True)

        # Iterate through each JSON file in the dataset type
        for json_file in tqdm.tqdm(os.listdir(dataset_path)):
            if json_file.endswith('.json'):
                json_path = dataset_path / json_file
                image_path = dataset_path / json_file.replace('.json', '.jpg')

                # New JSON and image paths
                new_json_path = keypoint_dir_path / json_file
                new_image_path = image_dir_path / json_file.replace('.json', '.jpg')

                # Load file and calculate keypoints
                calculator.load_soccernet_data(json_path)
                keypoints, lines = calculator.calculate_field_keypoints()
                with open(new_json_path, 'w') as f:
                    json.dump(keypoints, f)
                
                # Save annotated image
                calculator.visualize_keypoints(image_path, keypoints, lines, new_image_path)