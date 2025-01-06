import os
import pandas as pd
pd.options.mode.chained_assignment = None
import yaml
from tqdm import tqdm
from constants import dataset_dir

BASE_DIR = os.path.join(dataset_dir, 'tracking')
images_path = os.path.join(BASE_DIR, 'images')
labels_path = os.path.join(BASE_DIR, 'labels')

train_image_folders = []
val_image_folders = []
test_image_folders = []


def read_gt_file(ground_truth_path):
    """Processes GT File and return dataframe in yolov11 format"""

    # Read the data
    raw_gt_data = pd.read_csv(ground_truth_path, header=None)
    raw_gt_data.columns = ['frame', 'object', 'x', 'y', 'w', 'h', 'uk1', 'uk2', 'uk3', 'uk4']
    raw_gt_data['class'] = 0
    ground_truth_data = raw_gt_data.loc[:, ['class', 'x', 'y', 'w', 'h', 'frame']].copy()

    # Preprocess the data
    ground_truth_data['x'] = (ground_truth_data['x'] + (ground_truth_data['w'] / 2)) / 1920
    ground_truth_data['y'] = (ground_truth_data['y'] + (ground_truth_data['h'] / 2)) / 1080
    ground_truth_data['w'] = ground_truth_data['w'] / 1920
    ground_truth_data['h'] = ground_truth_data['h'] / 1080
    return ground_truth_data


if __name__ == "__main__":
    # Iterate through train/val/test
    for data_type in os.listdir(images_path):
        data_type_path = os.path.join(images_path, data_type)
        print(data_type_path)

        # Iterate through each clip
        for vid in tqdm(os.listdir(data_type_path)):
            vid_path = os.path.join(data_type_path, vid)
            image_folder_path = os.path.join(vid_path, 'img1')

            # Populate Folders
            if data_type == 'train':
                train_image_folders.append(image_folder_path.split(BASE_DIR)[-1][1:])
            elif data_type == 'test':
                val_image_folders.append(image_folder_path.split(BASE_DIR)[-1][1:])
            elif data_type == 'challenge':
                test_image_folders.append(image_folder_path.split(BASE_DIR)[-1][1:])

            # Process GT Data of train or val
            if data_type in ['train', 'test']:
                gt_path = os.path.join(vid_path, 'gt/gt.txt')
                gt_data = read_gt_file(gt_path)

                # For each image, filter out rows from gt_data and dump it in txt file
                for image in os.listdir(image_folder_path):
                    if '.jpg' in image:
                        image_number = int(image.split('.jpg')[0])
                        image_path = os.path.join(image_folder_path, image)

                        # Filter out rows from gt_data
                        gt_image_data = gt_data[gt_data['frame'] == image_number]
                        gt_image_data.drop('frame', axis=1, inplace=True)

                        # Dump it in txt file
                        image_label_path = image_path.replace(r'images', r"labels").replace(".jpg", ".txt")
                        image_label_base_dir = os.path.dirname(image_label_path)
                        os.makedirs(image_label_base_dir, exist_ok=True)
                        gt_image_data.to_csv(image_label_path, index=False, header=False, sep=' ')

    data = {'path': BASE_DIR,
            'train': train_image_folders,
            'val': val_image_folders,
            # 'test': test_image_folders,
            'names': {0: 'Present'}}

    # Dump YAML and path to YAML
    base_dir_yaml = os.path.join(BASE_DIR, 'data.yaml')
    with open(base_dir_yaml, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    with open('yaml_path.txt', 'w') as f:
        f.write(base_dir_yaml)
