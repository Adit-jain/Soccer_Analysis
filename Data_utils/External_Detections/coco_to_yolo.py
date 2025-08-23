"""COCO to YOLO format converter.

This script converts COCO format annotations to Ultralytics YOLO format,
supporting both bounding boxes and segmentation masks.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

def coco_to_yolo(coco_json_path: str, output_labels_dir: str, use_segments: bool = False) -> None:
    """Convert COCO format annotations to Ultralytics YOLO format.

    Args:
        coco_json_path: Path to the COCO JSON annotation file
        output_labels_dir: Directory to save the YOLO format label files
        use_segments: If True, convert segmentation polygons to YOLO format,
                     otherwise convert bounding boxes
    """
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create mapping from COCO category_id to 0-indexed YOLO class_id
    coco_cat_ids = sorted(categories.keys())
    coco_to_yolo_class_id = {coco_cat_id: i for i, coco_cat_id in enumerate(coco_cat_ids)}
    
    yolo_class_names = [categories[coco_cat_id] for coco_cat_id in coco_cat_ids]

    print(f"Found {len(images)} images and {len(categories)} categories.")
    print("Categories (COCO ID -> YOLO ID: Name):")
    for coco_id, yolo_id in coco_to_yolo_class_id.items():
        print(f"  {coco_id} -> {yolo_id}: {categories[coco_id]}")
    
    # Store class names in classes.txt file (YOLO best practice)
    classes_txt_path = output_labels_dir.parent / 'classes.txt'
    with open(classes_txt_path, 'w', encoding='utf-8') as f:
        for name in yolo_class_names:
            f.write(f"{name}\n")
    print(f"Saved class names to: {classes_txt_path}")

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Process each image and its annotations
    for img_id, img_data in tqdm(images.items(), desc="Converting annotations"):
        img_filename = img_data['file_name']
        img_width = img_data['width']
        img_height = img_data['height']

        yolo_annotations = []
        
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                coco_cat_id = ann['category_id']
                yolo_class_id = coco_to_yolo_class_id.get(coco_cat_id)

                if yolo_class_id is None:
                    print(f"Warning: Category ID {coco_cat_id} not found in categories. Skipping.")
                    continue

                if use_segments and 'segmentation' in ann and ann['segmentation']:
                    # Convert segmentation polygons
                    # Single object can have multiple polygons (if occluded)
                    # Taking first polygon for simplicity
                    # YOLO format: class_id x1 y1 x2 y2 ... xn yn (normalized)
                    # COCO format: [[x1,y1,x2,y2,x3,y3,...]] or RLE
                    
                    seg = ann['segmentation']
                    if isinstance(seg, list) and len(seg) > 0:
                        # Take first polygon
                        polygon = seg[0]
                        if not isinstance(polygon, list) or len(polygon) < 6:  # Need ≥3 points
                            continue  # Skip invalid polygons

                        normalized_polygon = []
                        for i in range(0, len(polygon), 2):
                            x = polygon[i] / img_width
                            y = polygon[i+1] / img_height
                            normalized_polygon.extend([x, y])
                        
                        yolo_annotations.append(f"{yolo_class_id} " + " ".join(map(str, normalized_polygon)))
                    # Note: RLE format not handled for simplicity
                
                elif not use_segments and 'bbox' in ann:
                    # Convert bounding box
                    # COCO bbox: [x_min, y_min, width, height] (absolute)
                    # YOLO bbox: <x_center_norm> <y_center_norm> <width_norm> <height_norm>
                    x_min, y_min, bbox_width, bbox_height = ann['bbox']

                    x_center = x_min + bbox_width / 2
                    y_center = y_min + bbox_height / 2

                    norm_x_center = x_center / img_width
                    norm_y_center = y_center / img_height
                    norm_width = bbox_width / img_width
                    norm_height = bbox_height / img_height
                    
                    yolo_annotations.append(
                        f"{yolo_class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )

        # Write YOLO label file (create empty file even if no annotations)
        label_filename_base = Path(img_filename).stem
        label_file_path = output_labels_dir / f"{label_filename_base}.txt"
        
        with open(label_file_path, 'w', encoding='utf-8') as f_out:
            for line in yolo_annotations:
                f_out.write(line + "\n")

    print(f"✓ Conversion complete. YOLO labels saved to: {output_labels_dir}")
    print("📝 Remember to create a data.yaml file for Ultralytics training.")

def main() -> None:
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert COCO JSON annotations to Ultralytics YOLO format."
    )
    parser.add_argument(
        "coco_json", 
        type=str, 
        help="Path to COCO JSON annotation file (e.g., instances_train2017.json)"
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Directory to save YOLO format labels (e.g., ./coco/labels/train2017)"
    )
    parser.add_argument(
        "--use_segments", 
        action='store_true', 
        help="Convert segmentation polygons instead of bounding boxes"
    )
    
    args = parser.parse_args()
    coco_to_yolo(args.coco_json, args.output_dir, args.use_segments)


if __name__ == '__main__':
    main()

