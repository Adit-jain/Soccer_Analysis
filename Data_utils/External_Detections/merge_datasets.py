"""Merge multiple COCO datasets with class mapping and image limits.

This script combines multiple COCO format datasets into a single dataset,
allowing for class name standardization and limiting images per dataset.
"""

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

random.seed(2)

def merge_coco_datasets(
    dataset_paths: List[str], 
    class_map: Dict[str, int], 
    image_limits: Dict[str, int], 
    output_path: str,
    final_id_map: Dict[int, str]
) -> None:
    """Merge multiple COCO datasets with class mapping and image limits.
    
    Args:
        dataset_paths: List of paths to COCO dataset JSON files
        class_map: Mapping of original class names to new IDs
        image_limits: Maximum images per dataset (-1 for unlimited)
        output_path: Path to save merged COCO JSON file
        final_id_map: Mapping from final class IDs to class names
    """
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {
            "description": "Merged Soccer Analysis Dataset",
            "version": "1.0",
            "contributor": "Soccer Analysis Project"
        }
    }
    
    class_name_to_id = {}
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_path in dataset_paths:
        # Try loading with different naming conventions
        paths_to_try = [
            dataset_path,
            dataset_path.replace('_annotations.coco.json', 'annotations_coco.json')
        ]
        
        coco_data = None
        for path in paths_to_try:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                dataset_path = path  # Update to successful path
                break
            except FileNotFoundError:
                continue
        
        if coco_data is None:
            print(f"Warning: Could not load dataset from {dataset_path}")
            continue
        
        dataset_dir = Path(dataset_path).parent
        dataset_name = Path(dataset_path).parts[-3]
        print(f"Processing dataset: {dataset_name}")
        max_images = image_limits.get(dataset_name, None)
        
        # Map categories
        category_mapping = {}
        for category in coco_data.get("categories", []):
            name = category["name"]
            if name in class_map:
                final_id = class_map[name]
                final_name = final_id_map[final_id]
                if final_name not in class_name_to_id:
                    class_name_to_id[final_name] = final_id
                category_mapping[category["id"]] = final_id

        # Select and process images
        image_id_map = {}
        available_images = coco_data.get("images", [])
        
        if max_images != -1 and len(available_images) > max_images:
            selected_images = random.sample(available_images, max_images)
        else:
            selected_images = available_images
        
        for new_image_id, image in enumerate(selected_images, start=len(merged_data["images"])):
            image_id_map[image["id"]] = new_image_id
            image["id"] = new_image_id
            
            # Copy image file with new naming
            current_path = dataset_dir / image["file_name"]
            new_filename = f"{dataset_name}_{new_image_id}.jpg"
            new_path = output_dir / new_filename
            
            try:
                shutil.copy(current_path, new_path)
                image['file_name'] = new_filename
                merged_data["images"].append(image)
            except FileNotFoundError:
                print(f"Warning: Image not found: {current_path}")
        
        # Process annotations
        for annotation in coco_data.get("annotations", []):
            if (annotation["image_id"] in image_id_map and 
                annotation["category_id"] in category_mapping):
                annotation["image_id"] = image_id_map[annotation["image_id"]]
                annotation["category_id"] = category_mapping[annotation["category_id"]]
                annotation["id"] = len(merged_data["annotations"])
                merged_data["annotations"].append(annotation)
    
    # Add categories to merged dataset
    merged_data["categories"] = [
        {"id": id_, "name": name, "supercategory": "object"} 
        for name, id_ in class_name_to_id.items()
    ]
    
    # Save merged dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"✓ Merged dataset saved at {output_path}")
    print(f"  - Total images: {len(merged_data['images'])}")
    print(f"  - Total annotations: {len(merged_data['annotations'])}")
    print(f"  - Classes: {list(class_name_to_id.keys())}")


def main() -> None:
    """Main function to execute dataset merging with predefined configurations."""
    # Dataset paths
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

    # Class mapping: original names -> standardized IDs
    class_map = {
        'player': 1, 'Player': 1, 'Team-A': 1, 'Team-H': 1,
        'football player': 1, 'goalkeeper': 1, 'Gardien': 1, 'Joueur': 1,
        'ball': 2, 'Ball': 2, 'Ballon': 2, 'football': 2,
        'referee': 3, 'Referee': 3, 'Arbitre': 3,
    }

    # Final class ID to name mapping
    final_id_map = {1: 'Player', 2: 'Ball', 3: 'Referee'}

    # Image limits per dataset (-1 = unlimited)
    image_limits = {
        "spt_v2": 30, "spt_v2_sahi_160": 30, "spt_v2_sahi_320": 40,
        "tbd_v2": -1, "v2_temp": 300, "v2_temp_sahi_160": 300,
        "v2_temp_sahi_320": 400, "v3": 500, "v3_sahi_160": 500,
        "v3_sahi_320": 1000, "v3_sahi_640": 500, "v5_temp": 500,
        "v7": 500, "v7_sahi_160": 500, "v7_sahi_320": 1000,
        "v7_sahi_640": 500, "v12": 200, "v12_sahi_160": 300,
        "v12_sahi_320": 500, "v12_sahi_640": 300,
    }

    output_path = r"D:\Datasets\SoccerAnalysis_Final\V1/train/_annotations.coco.json"
    
    merge_coco_datasets(dataset_paths, class_map, image_limits, output_path, final_id_map)


if __name__ == '__main__':
    main()