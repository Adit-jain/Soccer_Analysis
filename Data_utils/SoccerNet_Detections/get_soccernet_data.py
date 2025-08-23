"""SoccerNet tracking data downloader.

Downloads SoccerNet tracking dataset with multi-object tracking annotations
for players, ball, and other objects in soccer match videos.
"""

from typing import List

from constants import dataset_dir, soccernet_password
from SoccerNet.Downloader import SoccerNetDownloader


def download_tracking_data(
    local_directory: str = dataset_dir,
    password: str = soccernet_password,
    splits: List[str] = ["train", "test", "challenge"],
    task: str = "tracking"
) -> None:
    """Download SoccerNet tracking dataset.
    
    Args:
        local_directory: Local directory to save downloaded data
        password: SoccerNet API password
        splits: Dataset splits to download
        task: Tracking task to download
    """
    print("📎 Initializing SoccerNet tracking downloader...")
    
    # Initialize SoccerNet downloader
    downloader = SoccerNetDownloader(LocalDirectory=local_directory)
    downloader.password = password
    
    print(f"  Local directory: {local_directory}")
    print(f"  Task: {task}")
    print(f"  Splits: {splits}")
    
    # Download tracking data
    print(f"\n📁 Downloading {task} dataset...")
    try:
        downloader.downloadDataTask(task=task, split=splits)
        print(f"  ✓ Successfully downloaded {task} data")
        print(f"  📋 Dataset includes: players, ball tracking annotations")
        print(f"  🎥 Video sequences with MOT format ground truth")
    except Exception as e:
        print(f"  ❌ Error downloading {task}: {e}")
        raise
    
    print(f"\n🎉 Tracking data download completed!")


def main() -> None:
    """Main function to execute tracking data download."""
    try:
        download_tracking_data()
        print(f"\n📄 Next steps:")
        print(f"  1. Run data_preprocessing.py to convert to YOLO format")
        print(f"  2. Use processed data for model training")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        raise


if __name__ == "__main__":
    main()