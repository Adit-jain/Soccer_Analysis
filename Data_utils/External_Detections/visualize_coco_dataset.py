"""COCO dataset visualization tool.

Utility script to visualize COCO format annotations with bounding boxes
and category labels overlaid on images.
"""

import os
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO

def visualize_coco_dataset(
    image_dir: str, 
    annotation_file: str, 
    num_samples: int = 10,
    save_dir: Optional[str] = None,
    show_images: bool = True
) -> None:
    """Visualize images and annotations from a COCO dataset.
    
    Args:
        image_dir: Path to directory containing images
        annotation_file: Path to COCO annotation JSON file
        num_samples: Number of random images to visualize
        save_dir: Directory to save visualization images (optional)
        show_images: Whether to display images interactively
    """
    try:
        # Load COCO dataset
        coco = COCO(annotation_file)
        print(f"📎 Loaded COCO dataset with {len(coco.getImgIds())} images")
        
        # Get all image IDs and sample random subset
        img_ids = coco.getImgIds()
        if len(img_ids) < num_samples:
            print(f"Warning: Only {len(img_ids)} images available, using all")
            selected_ids = img_ids
        else:
            selected_ids = random.sample(img_ids, num_samples)
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
    except Exception as e:
        print(f"Error loading COCO dataset: {e}")
        return
    
    for idx, img_id in enumerate(selected_ids, 1):
        try:
            img_info = coco.loadImgs(img_id)[0]
            img_path = Path(image_dir) / img_info['file_name']
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
                
            # Load image
            img = Image.open(img_path)
            
            # Get annotations
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # Create visualization
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
            
            # Draw bounding boxes and labels
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
            for ann in anns:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                category_info = coco.loadCats(category_id)[0]
                category_name = category_info['name']
                
                # Use different colors for different categories
                color = colors[category_id % len(colors)]
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(
                    bbox[0], bbox[1] - 5, 
                    f"{category_name} (ID: {category_id})",
                    color='white', fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
                )
            
            # Set title and remove axes
            ax.set_title(
                f"Image {idx}/{len(selected_ids)}: {img_info['file_name']} "
                f"({len(anns)} annotations)",
                fontsize=14, weight='bold'
            )
            ax.axis('off')
            
            # Save or show image
            if save_dir:
                save_path = Path(save_dir) / f"visualization_{idx:03d}_{img_info['file_name']}"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {save_path}")
                
            if show_images:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            continue

def main() -> None:
    """Main function with example usage."""
    # Example configurations
    image_dir = r"F:\Datasets\SoccerAnalysis_Final\V1\images\train"
    annotation_file = r"F:\Datasets\SoccerAnalysis_Final\V1\coco_train_annotations\_annotations.coco.json"
    
    print("🎨 Starting COCO dataset visualization...")
    
    # Visualize dataset
    visualize_coco_dataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        num_samples=5,  # Show 5 random images
        save_dir=None,  # Set to a path to save images
        show_images=True
    )
    
    print("✓ Visualization completed!")


if __name__ == '__main__':
    main()
