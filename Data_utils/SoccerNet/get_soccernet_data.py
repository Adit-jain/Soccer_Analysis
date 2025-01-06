from constants import dataset_dir, soccernet_password

# Initialize SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=dataset_dir)
mySoccerNetDownloader.password = soccernet_password

# Download Tracking Labels for 2022
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])