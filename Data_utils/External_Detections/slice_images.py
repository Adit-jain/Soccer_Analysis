"""SAHI-based image slicing for COCO datasets.

This script uses SAHI (Slicing Aided Hyper Inference) to slice large images
into smaller patches with multiple slice sizes for better object detection.
"""

from pathlib import Path
from typing import List, Tuple
from sahi.slicing import slice_coco


def slice_datasets(
    dataset_paths: List[str], 
    output_folders: List[str], 
    slice_sizes: List[int] = [160, 320, 640],
    splits: List[str] = ['train', 'valid', 'test']
) -> None:
    """Slice multiple COCO datasets at different scales using SAHI.
    
    Args:
        dataset_paths: List of paths to COCO dataset directories
        output_folders: List of output directory names (without size suffix)
        slice_sizes: List of slice dimensions (width x height)
        splits: Dataset splits to process
    """
    for dataset_path, output_base in zip(dataset_paths, output_folders):
        dataset_name = Path(dataset_path).name
        print(f"\n📎 Processing dataset: {dataset_name}")
        
        for split in splits:
            annotation_file = Path(dataset_path) / split / "_annotations.coco.json"
            image_dir = Path(dataset_path) / split
            
            # Skip if annotation file doesn't exist
            if not annotation_file.exists():
                print(f"  ⚠️  Skipping {split}: annotation file not found")
                continue
                
            for size in slice_sizes:
                output_dir = f"{output_base}_{size}/{split}"
                print(f"  🔪 Slicing {split} split with {size}x{size} patches")
                
                try:
                    slice_coco(
                        coco_annotation_file_path=str(annotation_file),
                        image_dir=str(image_dir),
                        output_coco_annotation_file_name="annotations",
                        output_dir=output_dir,
                        slice_width=size,
                        slice_height=size,
                        overlap_ratio=0.2  # 20% overlap between patches
                    )
                    print(f"    ✓ Completed {size}x{size} slicing for {split}")
                except Exception as e:
                    print(f"    ❌ Failed {size}x{size} slicing for {split}: {e}")


def main() -> None:
    """Main function with predefined dataset configurations."""
    # Dataset configurations
    datasets = [
        r"D:\Datasets\SoccerAnalysis\Player Detection.v3i.coco",
        r"D:\Datasets\SoccerAnalysis\VA_Project.v2i.coco",
    ]

    output_folders = [
        r"D:\Datasets\SoccerAnalysis\v3_sahi",
        r"D:\Datasets\SoccerAnalysis\v2_temp_sahi"
    ]
    
    # Execute slicing
    slice_datasets(datasets, output_folders)
    print("\n🎉 All dataset slicing completed!")


if __name__ == '__main__':
    main()
