"""SoccerNet JSON file transfer utility.

Utility to consolidate SoccerNet calibration JSON files from different
splits (train/test/valid) into a unified directory structure.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

def transfer_json_files(
    source_dir: str = r"F:\Datasets\SoccerNet\Data\calibration",
    destination_dir: Optional[str] = None,
    copy_mode: bool = True,
    splits: List[str] = ["train", "test", "valid"]
) -> None:
    """Transfer JSON files from SoccerNet calibration dataset splits.
    
    Consolidates JSON annotation files from different dataset splits into
    a unified directory structure for easier processing.
    
    Args:
        source_dir: Source directory containing split folders
        destination_dir: Destination directory (if None, creates subfolder in source_dir)
        copy_mode: If True, copy files. If False, move files
        splits: List of dataset splits to process
    """
    print(f"📎 Starting JSON file transfer...")
    print(f"  Source: {source_dir}")
    print(f"  Mode: {'Copy' if copy_mode else 'Move'}")
    
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"\u274c Error: Source directory not found: {source_path}")
        return
    
    # Determine destination path
    if destination_dir is None:
        dest_path = source_path / "SoccerNet_Calibration_JSON"
    else:
        dest_path = Path(destination_dir)
    
    print(f"  Destination: {dest_path}")
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    total_files = 0
    transferred_files = 0
    errors = []
    
    # Process each dataset split
    for split in splits:
        split_path = source_path / split
        
        if not split_path.exists():
            print(f"  ⚠️ Warning: {split} folder not found at {split_path}")
            continue
        
        if not split_path.is_dir():
            print(f"  ⚠️ Warning: {split_path} is not a directory")
            continue
            
        # Create destination subfolder
        dest_split_folder = dest_path / split
        dest_split_folder.mkdir(exist_ok=True)
        
        # Find all JSON files
        json_files = list(split_path.glob("*.json"))
        total_files += len(json_files)
        
        print(f"\n📁 Processing {len(json_files)} JSON files from {split}/")
        
        # Transfer each JSON file
        for json_file in json_files:
            dest_file = dest_split_folder / json_file.name
            
            try:
                if dest_file.exists() and not copy_mode:
                    print(f"    ⚠️ Skipping {json_file.name} (destination exists)")
                    continue
                    
                if copy_mode:
                    shutil.copy2(json_file, dest_file)
                else:
                    shutil.move(str(json_file), str(dest_file))
                
                transferred_files += 1
                
                # Show progress for large transfers
                if transferred_files % 100 == 0:
                    print(f"    Processed {transferred_files} files...")
                    
            except Exception as e:
                error_msg = f"Error transferring {json_file.name}: {e}"
                print(f"    ❌ {error_msg}")
                errors.append(error_msg)
    
    # Print transfer summary
    print(f"\n🎉 Transfer complete!")
    print(f"  📄 Total JSON files found: {total_files}")
    print(f"  ✓ Successfully transferred: {transferred_files}")
    print(f"  📂 Files transferred to: {dest_path}")
    
    if errors:
        print(f"  ⚠️ Errors encountered: {len(errors)}")
        for error in errors:
            print(f"    - {error}")
    
    # Verify transfer
    transferred_total = sum(len(list((dest_path / split).glob("*.json"))) 
                          for split in splits if (dest_path / split).exists())
    print(f"  🔍 Verification: {transferred_total} JSON files in destination")

def batch_transfer_multiple_sources(
    source_configs: List[dict],
    base_destination: str
) -> None:
    """Transfer from multiple source directories.
    
    Args:
        source_configs: List of config dicts with 'source_dir' and optional params
        base_destination: Base destination directory
    """
    for i, config in enumerate(source_configs, 1):
        print(f"\n=== Processing source {i}/{len(source_configs)} ===")
        dest_subdir = Path(base_destination) / f"source_{i}"
        
        transfer_json_files(
            source_dir=config['source_dir'],
            destination_dir=str(dest_subdir),
            copy_mode=config.get('copy_mode', True),
            splits=config.get('splits', ["train", "test", "valid"])
        )


def main() -> None:
    """Main function with example usage scenarios."""
    print("SoccerNet JSON Transfer Utility")
    print("=" * 40)
    
    # Default behavior: copy all JSON files to unified folder
    transfer_json_files()
    
    # Alternative usage examples (commented out):
    
    # Example 1: Transfer to custom destination
    # transfer_json_files(
    #     destination_dir=r"D:\Projects\Soccer_Analysis\Data\json_files",
    #     copy_mode=True
    # )
    
    # Example 2: Move files instead of copy
    # transfer_json_files(copy_mode=False)
    
    # Example 3: Process only specific splits
    # transfer_json_files(
    #     splits=["train", "test"],  # Skip 'valid'
    #     copy_mode=True
    # )
    
    # Example 4: Batch process multiple source directories
    # source_configs = [
    #     {'source_dir': r"F:\Datasets\SoccerNet\Data\calibration"},
    #     {'source_dir': r"F:\Datasets\SoccerNet\Data\calibration_v2", 'copy_mode': False}
    # ]
    # batch_transfer_multiple_sources(
    #     source_configs, 
    #     r"D:\Projects\Soccer_Analysis\Data\unified_json"
    # )


if __name__ == "__main__":
    main()