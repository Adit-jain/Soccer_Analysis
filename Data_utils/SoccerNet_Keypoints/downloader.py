from constants import dataset_dir, soccernet_password

# Initialize SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=dataset_dir)
mySoccerNetDownloader.password = soccernet_password

mySoccerNetDownloader.downloadDataTask(task="calibration", split=["train","valid","test"]) 
mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train","valid","test"])