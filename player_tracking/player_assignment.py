import supervision as sv
from transformers import AutoProcessor, SiglipVisionModel
import torch
import tqdm
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_assignment_models():
    """
    Initialize the assignment models for the player tracking system.
    """
    model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    reducer = umap.UMAP(n_components=3)
    cluster_model = KMeans(n_clusters=3)

    return model, processor, reducer, cluster_model


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


def get_siglip_embeddings(cropped_images, model, processor):
    """
    Get the embeddings of the cropped images using the Siglip model.
    """
    data = []
    with torch.no_grad():
        for batch in tqdm.tqdm([cropped_images]):
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)
    data = np.concatenate(data, axis=0)
    return data


def project_embeddings(data, reducer):
    """
    Project the embeddings into a lower-dimensional space.
    """
    return reducer.fit_transform(data)


def cluster_embeddings(data, cluster_model):
    """
    Cluster the embeddings using a KMeans model.
    """
    return cluster_model.fit_predict(data)


def assign_players(frame, player_detections, model, processor, reducer, cluster_model):
    """
    Assign the players to their respective teams.
    """
    
    cropped_images = get_player_crops(frame, player_detections)
    embeddings = get_siglip_embeddings(cropped_images, model, processor)
    embeddings = project_embeddings(embeddings, reducer)
    assignments = cluster_embeddings(embeddings, cluster_model)
    return assignments


    