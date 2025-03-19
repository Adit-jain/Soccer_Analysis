import json
import random
import os
from collections import defaultdict
import shutil
random.seed(2)

def merge_coco_datasets(dataset_paths, class_map, image_limits, output_path):
    """
    Merges multiple COCO datasets while selecting specific classes and limiting the number of images.
    
    Parameters:
    - dataset_paths (list of str): Paths to the COCO dataset JSON files.
    - class_map (dict): Mapping of class names to keep {"class_name": new_id}.
    - image_limits (dict): Limits on the number of images per dataset {"dataset_name": max_images}.
    - output_path (str): Path to save the merged COCO JSON file.
    """
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    
    class_name_to_id = {}
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    for dataset_path in dataset_paths:
        try:
            with open(dataset_path, 'r') as f:
                coco_data = json.load(f)
        except FileNotFoundError:
            dataset_path = dataset_path.replace('_annotations.coco.json', 'annotations_coco.json')
            with open(dataset_path, 'r') as f:
                coco_data = json.load(f)
        
        dataset_dir = os.path.dirname(dataset_path)
        dataset_name = dataset_path.split('/')[-3]
        print(dataset_name)
        max_images = image_limits.get(dataset_name, None)
        
        # Map categories
        category_mapping = {}
        for category in coco_data["categories"]:
            name = category["name"]
            if name in class_map:
                final_id = class_map[name]
                final_name = final_id_map[final_id]
                if final_name not in class_name_to_id:
                    class_name_to_id[final_name] = final_id
                category_mapping[category["id"]] = final_id

        # Filter images and annotations
        image_id_map = {}
        if max_images != -1:
            selected_images = random.sample(coco_data["images"], min(len(coco_data["images"]), max_images))
        else:
            selected_images = coco_data["images"]
        
        for new_image_id, image in enumerate(selected_images, start=len(merged_data["images"])):
            image_id_map[image["id"]] = new_image_id
            image["id"] = new_image_id
            current_path = os.path.join(dataset_dir, image["file_name"])
            new_filename = f"{dataset_name}_{new_image_id}.jpg"
            new_path = os.path.join(output_dir, new_filename)
            image['file_name'] = new_filename
            shutil.copy(current_path, new_path)

            merged_data["images"].append(image)
        
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] in image_id_map and annotation["category_id"] in category_mapping:
                annotation["image_id"] = image_id_map[annotation["image_id"]]
                annotation["category_id"] = category_mapping[annotation["category_id"]]
                annotation["id"] = len(merged_data["annotations"])
                merged_data["annotations"].append(annotation)
    
    # Add categories to the merged dataset
    merged_data["categories"] = [{"id": id_, "name": name} for name, id_ in class_name_to_id.items()]
    
    # Save the merged dataset
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"Merged dataset saved at {output_path}")


# List of dataset JSON paths
dataset_paths = [
 r'D:/Datasets/SoccerAnalysis/spt_v2/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/spt_v2_sahi_160/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/spt_v2_sahi_320/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/tbd_v2/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v12/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v12_sahi_160/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v12_sahi_320/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v12_sahi_640/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v2_temp/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v2_temp_sahi_160/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v2_temp_sahi_320/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v3/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v3_sahi_160/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v3_sahi_320/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v3_sahi_640/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v5_temp/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v7/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v7_sahi_160/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v7_sahi_320/train/_annotations.coco.json',
 r'D:/Datasets/SoccerAnalysis/v7_sahi_640/train/_annotations.coco.json',
]

# Mapping of selected classes to new category IDs
class_map = {
    'player': 1,
    'Player': 1,
    'Team-A': 1,
    'Team-H': 1,
    'football player': 1,
    'goalkeeper': 1,
    'Gardien': 1,
    'Joueur': 1,
    'ball': 2,
    'Ball': 2,
    'Ballon': 2,
    'football': 2,
    'referee': 3,
    'Referee': 3,
    'Arbitre': 3,
}

final_id_map = {
    1 : 'Player',
    2 : 'Ball',
    3 : 'Referee'
}

# Maximum number of images to take from each dataset
image_limits = {
    "spt_v2": 30,
    "spt_v2_sahi_160" : 30,
    "spt_v2_sahi_320" : 40,
    "tbd_v2" : -1,
    "v2_temp": 300,
    "v2_temp_sahi_160": 300,
    "v2_temp_sahi_320": 400,
    "v3": 500,
    "v3_sahi_160": 500,
    "v3_sahi_320": 1000,
    "v3_sahi_640": 500,
    "v5_temp": 500,
    "v7": 500,
    "v7_sahi_160": 500,
    "v7_sahi_320": 1000,
    "v7_sahi_640": 500,
    "v12": 200,
    "v12_sahi_160": 300,
    "v12_sahi_320": 500,
    "v12_sahi_640": 300,
}

# Output file path
output_path = r"D:\Datasets\SoccerAnalysis_Final\V1/train/_annotations.coco.json"

# Call the function
merge_coco_datasets(dataset_paths, class_map, image_limits, output_path)