"""SoccerNet calibration data downloader.

Downloads SoccerNet calibration dataset including field line annotations
and camera calibration data for keypoint extraction and pose estimation.
"""

from typing import List

from constants import dataset_dir, soccernet_password
from SoccerNet.Downloader import SoccerNetDownloader


def download_calibration_data(
    local_directory: str = dataset_dir,
    password: str = soccernet_password,
    splits: List[str] = ["train", "valid", "test"],
    tasks: List[str] = ["calibration", "calibration-2023"]
) -> None:
    """Download SoccerNet calibration datasets.
    
    Args:
        local_directory: Local directory to save downloaded data
        password: SoccerNet API password
        splits: Dataset splits to download
        tasks: Calibration tasks to download
    """
    print("📎 Initializing SoccerNet downloader...")
    
    # Initialize SoccerNet downloader
    downloader = SoccerNetDownloader(LocalDirectory=local_directory)
    downloader.password = password
    
    print(f"  Local directory: {local_directory}")
    print(f"  Splits: {splits}")
    print(f"  Tasks: {tasks}")
    
    # Download each calibration task
    for task in tasks:
        print(f"\n📁 Downloading {task} data...")
        try:
            downloader.downloadDataTask(task=task, split=splits)
            print(f"  ✓ Successfully downloaded {task}")
        except Exception as e:
            print(f"  ❌ Error downloading {task}: {e}")
    
    print(f"\n🎉 Calibration data download completed!")


def main() -> None:
    """Main function to execute calibration data download."""
    try:
        download_calibration_data()
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        raise


if __name__ == "__main__":
    main()