
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv
from ultralytics import YOLO
from player_detection import load_detection_model, get_detections
from player_tracking.tracking_constants import model_path, test_video
from player_tracking.assign_players import init_assignment_models, train_umap_kmeans, get_cluster_labels
from player_tracking.track_players import init_tracking_models, update_player_detections, annotate_players
import cv2
import time
from tqdm import tqdm


def annotation_callback(frame: np.ndarray, detection_model, siglip_model, siglip_processor, reducer, cluster_model, 
                        tracker, ellipse_annotator, traingle_annotator, label_annotator) -> np.ndarray:
    """
    A callback function to annotate the frames ellipses, labels, and team assignments.
    """

    # Get Detections
    detection_time = time.time()
    player_detections, ball_detections, referee_detections = get_detections(detection_model, frame)
    detection_time = time.time() - detection_time
    print(f"Detection Time: {detection_time:.2f}s")
    
    # In case any players are detected
    if len(player_detections) > 0:
        
        # Get Team Assignments
        assignment_time = time.time()
        clustered_embeddings = get_cluster_labels(frame, player_detections, siglip_model, siglip_processor, reducer, cluster_model)
        print(clustered_embeddings)
        assignment_time = time.time() - assignment_time
        print(f"Assignment Time: {assignment_time:.2f}s")

        # Update Player Tracking
        player_detections = update_player_detections(player_detections, tracker)
        
        # Assign Teams
        player_detections.class_id = clustered_embeddings
        ball_detections.class_id += 1
        referee_detections.class_id += 1
        print(player_detections)
        print(ball_detections)
        print(referee_detections)

    # Annotate the frame
    labeled_frame = annotate_players(frame, player_detections, ball_detections, referee_detections, ellipse_annotator, traingle_annotator, label_annotator)

    return labeled_frame


def track_players_realtime(video_path, model_path):
    """
    This function detects players in a video in real-time
    """
    # Load the Model
    model_init_time = time.time()
    print("Initializing the model...")
    detection_model = load_detection_model(model_path)

    # Initialize the annotators
    print("Initializing the annotators...")
    tracker, ellipse_annotator, traingle_annotator, label_annotator = init_tracking_models()

    # Initialize the Player Assignment Models
    print("Initializing the player assignment models...")
    siglip_model, siglip_processor, reducer, cluster_model = init_assignment_models()
    model_init_time = time.time() - model_init_time
    print(f"Model Initialization Time: {model_init_time:.2f}s")

    # Train the UMAP and KMeans models
    print("Training the UMAP and KMeans models...")
    training_kmeans_time = time.time()
    _, reducer, cluster_model = train_umap_kmeans(video_path, detection_model, siglip_model, siglip_processor, reducer, cluster_model)
    training_kmeans_time = time.time() - training_kmeans_time
    print(f"Training KMeans Time: {training_kmeans_time:.2f}s")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if success:

            # Annotate the frame
            annotation_time = time.time()
            result = annotation_callback(frame, detection_model, siglip_model, siglip_processor, reducer, cluster_model, tracker, ellipse_annotator, traingle_annotator, label_annotator)
            annotation_time = time.time() - annotation_time
            print(f"Annotation Time: {annotation_time:.2f}s")

            # Display the frame
            cv2.imshow("YOLO Inference", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_players_realtime(test_video, model_path)