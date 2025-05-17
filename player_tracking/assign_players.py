import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_detection import get_detections
import supervision as sv
from transformers import AutoProcessor, SiglipVisionModel
import torch
from tqdm import tqdm
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans
from more_itertools import chunked

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_assignment_models():
    """
    Initialize the assignment models for the player tracking system.
    """
    siglip_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    reducer = umap.UMAP(n_components=3)
    cluster_model = KMeans(n_clusters=2)

    return siglip_model, siglip_processor, reducer, cluster_model


def get_player_crops(frame, player_detections):
    """
    Process a frame of the video to get the player assignments.
    """
    cropped_images = []
    for boxes in player_detections.xyxy:
        cropped_image = sv.crop_image(frame, boxes)
        cropped_images.append(cropped_image)
    cropped_images = [sv.cv2_to_pillow(cropped_image) for cropped_image in cropped_images]
    return cropped_images


def create_batches(data, batch_size):
    """
    Create batches of data.
    """
    return list(chunked(data, batch_size))


def get_siglip_embeddings(batches, siglip_model, siglip_processor):
    """
    Get the embeddings of the cropped images using the Siglip model.
    """
    data = []
    with torch.no_grad():
        for batch in tqdm(batches, desc='getting_embeddings'):
            inputs = siglip_processor(images=batch, return_tensors="pt").to(device)
            outputs = siglip_model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)
    data = np.concatenate(data, axis=0)
    return data


def project_embeddings(data, reducer, train=False):
    """
    Project the embeddings into a lower-dimensional space.
    """
    if train:
        reduced_embeddings = reducer.fit_transform(data)
    else:
        reduced_embeddings = reducer.transform(data)

    return reduced_embeddings, reducer


def cluster_embeddings(data, cluster_model, train=False):
    """
    Cluster the embeddings using a KMeans model.
    """
    if train:
        clustered_embeddings = cluster_model.fit_predict(data)
    else:
        clustered_embeddings = cluster_model.predict(data)

    return clustered_embeddings, cluster_model


def assign_batch(crop_batches, siglip_model, siglip_processor, reducer, cluster_model, train=False):
    """
    Assign the players to their respective teams.
    """
    crop_embeddings = get_siglip_embeddings(crop_batches, siglip_model, siglip_processor)
    reduced_embeddings, reducer = project_embeddings(crop_embeddings, reducer, train=train)
    clustered_embeddings, cluster_model = cluster_embeddings(reduced_embeddings, cluster_model, train=train)
    return clustered_embeddings, reducer, cluster_model


def train_umap_kmeans(VIDEO_PATH, detection_model, siglip_model, siglip_processor, reducer, cluster_model, crops=None):
    """
    Train the UMAP and KMeans models.
    """

    if crops is None:
        # Get the video frames
        frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=24, end=60*24)
        
        # Get player crops
        crops = []
        for frame in tqdm(frame_generator, desc='collecting_crops'):
            player_detections, _, _ = get_detections(detection_model, frame)
            cropped_images = get_player_crops(frame, player_detections)
            crops += cropped_images

    # Train and get assignments
    crop_batches = create_batches(crops, 24)
    clustered_embeddings, reducer, cluster_model = assign_batch(crop_batches, siglip_model, siglip_processor, reducer, cluster_model, train=True)
    return clustered_embeddings, reducer, cluster_model


def get_cluster_labels(frame, player_detections, siglip_model, siglip_processor, reducer, cluster_model, crops=None):
    """
    Get the cluster labels for the players.
    """

    if crops is None:
        # Get player crops
        crops = get_player_crops(frame, player_detections)
    
    # Get assignments
    clustered_embeddings, _, _ = assign_batch([crops], siglip_model, siglip_processor, reducer, cluster_model, train=False)

    return clustered_embeddings

    