import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import supervision as sv
from transformers import AutoProcessor, SiglipVisionModel
import torch
import numpy as np
from tqdm import tqdm
from more_itertools import chunked

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EmbeddingExtractor:
    """
    Handles extraction of visual embeddings from player crops using SigLIP model.
    """
    
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        """
        Initialize the embedding extractor with SigLIP model.
        
        Args:
            model_name: HuggingFace model name for SigLIP
        """
        self.model = SiglipVisionModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def get_player_crops(self, frame, player_detections):
        """
        Extract player crops from frame using detection bounding boxes.
        
        Args:
            frame: Input video frame
            player_detections: Player detection results
            
        Returns:
            List of cropped player images in PIL format
        """
        cropped_images = []
        for boxes in player_detections.xyxy:
            cropped_image = sv.crop_image(frame, boxes)
            cropped_images.append(cropped_image)
        cropped_images = [sv.cv2_to_pillow(cropped_image) for cropped_image in cropped_images]
        return cropped_images
    
    def create_batches(self, data, batch_size=24):
        """
        Create batches from data for efficient processing.
        
        Args:
            data: Input data to batch
            batch_size: Size of each batch
            
        Returns:
            List of batched data
        """
        return list(chunked(data, batch_size))
    
    def get_embeddings(self, image_batches):
        """
        Extract SigLIP embeddings from image batches.
        
        Args:
            image_batches: Batched images for processing
            
        Returns:
            Numpy array of embeddings
        """
        data = []
        with torch.no_grad():
            for batch in tqdm(image_batches, desc='extracting_embeddings'):
                inputs = self.processor(images=batch, return_tensors="pt").to(device)
                outputs = self.model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        data = np.concatenate(data, axis=0)
        return data