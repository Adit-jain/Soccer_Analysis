import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from PIL import Image

def visualize_coco(image_dir, annotation_file):
    """
    Visualizes images and annotations from a COCO dataset.
    
    :param image_dir: Path to the directory containing images
    :param annotation_file: Path to the COCO annotation JSON file
    :param num_samples: Number of images to visualize
    """
    # Load COCO dataset
    coco = COCO(annotation_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    selected_ids = img_ids
    
    for img_id in random.sample(selected_ids, 10):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        # Open image
        img = Image.open(img_path)
        
        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Plot image
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img)
        
        # Draw bounding boxes
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 5, category_name, color='white',
                    fontsize=12, backgroundcolor='red')
        
        plt.axis('off')
        plt.show()

# Example usage
visualize_coco(r"F:\Datasets\SoccerAnalysis_Final\V1\images\train", 
               r"F:\Datasets\SoccerAnalysis_Final\V1\coco_train_annotations\_annotations.coco.json")
