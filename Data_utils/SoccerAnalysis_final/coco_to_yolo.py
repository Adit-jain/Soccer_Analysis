import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def coco_to_yolo(coco_json_path, output_labels_dir, use_segments=False):
    """
    Converts COCO format annotations to Ultralytics YOLO format.

    Args:
        coco_json_path (str): Path to the COCO JSON annotation file.
        output_labels_dir (str): Directory to save the YOLO format label files.
        use_segments (bool): If True, convert segmentation polygons to YOLO format.
                             If False, convert bounding boxes.
    """
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create a mapping from COCO category_id to a 0-indexed YOLO class_id
    coco_cat_ids = sorted(categories.keys())
    coco_to_yolo_class_id = {coco_cat_id: i for i, coco_cat_id in enumerate(coco_cat_ids)}
    
    yolo_class_names = [categories[coco_cat_id] for coco_cat_id in coco_cat_ids]

    print(f"Found {len(images)} images and {len(categories)} categories.")
    print("Categories (COCO ID -> YOLO ID: Name):")
    for coco_id, yolo_id in coco_to_yolo_class_id.items():
        print(f"  {coco_id} -> {yolo_id}: {categories[coco_id]}")
    
    # Store class names in a classes.txt file (optional, but good practice for YOLO)
    classes_txt_path = output_labels_dir.parent / 'classes.txt' # Often placed alongside train/val folders
    with open(classes_txt_path, 'w') as f:
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
                    print(f"Warning: Category ID {coco_cat_id} not found in categories. Skipping annotation.")
                    continue

                if use_segments and 'segmentation' in ann and ann['segmentation']:
                    # Convert segmentation polygons
                    # A single object can have multiple polygons (e.g., if occluded)
                    # We take the first polygon for simplicity here.
                    # YOLO format for segments: class_id x1 y1 x2 y2 ... xn yn (all normalized)
                    # COCO segmentation: [[x1,y1,x2,y2,x3,y3,...]] or RLE
                    
                    seg = ann['segmentation']
                    if isinstance(seg, list) and len(seg) > 0:
                        # Take the first polygon
                        polygon = seg[0] 
                        if not isinstance(polygon, list) or len(polygon) < 6: # Need at least 3 points
                            # print(f"Warning: Skipping invalid polygon for image {img_filename}, ann_id {ann.get('id')}")
                            continue # Skip if not a valid polygon list

                        normalized_polygon = []
                        for i in range(0, len(polygon), 2):
                            x = polygon[i] / img_width
                            y = polygon[i+1] / img_height
                            normalized_polygon.extend([x, y])
                        
                        yolo_annotations.append(f"{yolo_class_id} " + " ".join(map(str, normalized_polygon)))
                    # else: RLE or empty segmentation, not handled for simplicity
                
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

        # Write YOLO label file for this image
        # Even if there are no annotations, an empty file should be created
        label_filename_base = Path(img_filename).stem
        label_file_path = output_labels_dir / f"{label_filename_base}.txt"
        
        with open(label_file_path, 'w') as f_out:
            for line in yolo_annotations:
                f_out.write(line + "\n")

    print(f"Conversion complete. YOLO labels saved to: {output_labels_dir}")
    print(f"Remember to create a data.yaml file for training with Ultralytics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to Ultralytics YOLO format.")
    parser.add_argument("coco_json", type=str, help="Path to the COCO JSON annotation file (e.g., instances_train2017.json).")
    parser.add_argument("output_dir", type=str, help="Directory to save the YOLO format label files (e.g., ./coco/labels/train2017).")
    parser.add_argument("--use_segments", action='store_true', help="Convert segmentation data instead of bounding boxes. If annotations contain polygons.")
    
    args = parser.parse_args()

    # Example Usage:
    # python coco_to_yolo.py path/to/your/coco_annotations.json path/to/output/yolo_labels --use_segments (optional)

    coco_to_yolo(args.coco_json, args.output_dir, args.use_segments)

    # --- Example data.yaml structure (you'll need to create this manually or script it) ---
    # train: ../coco/images/train2017  # path to train images
    # val: ../coco/images/val2017    # path to val images
    # test: ../coco/images/test2017   # path to test images (optional)
    #
    # # Classes
    # nc: 80  # number of classes
    # names: ['person', 'bicycle', ..., 'toothbrush'] # From your classes.txt or directly
    # --------------------------------------------------------------------------------------