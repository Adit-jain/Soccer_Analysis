import yaml
import os

data = {'path': r'D:\Datasets\SoccerAnalysis_Final\V1',
        'train': r'images/train',
        'val': r'images/test',
        'names': {0: 'Player', 1: 'Ball', 2: 'Referee'}}

yaml_path = os.path.join(r'D:\Datasets\SoccerAnalysis_Final\V1', 'data.yaml')
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)