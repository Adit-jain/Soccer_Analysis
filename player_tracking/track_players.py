try:
    from tracking_constants import model_path, test_video, PROJECT_PATH
except ImportError:
    from player_tracking.tracking_constants import model_path, test_video, PROJECT_PATH

import sys
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import numpy as np
import supervision as sv
from ultralytics import YOLO
from player_detection import load_detection_model, detect_players_in_frames
from player_tracking.player_assignment import init_assignment_models, get_player_crops, create_batches, assign_batch
import cv2
import time
from tqdm import tqdm


def get_detections(detection_model, frame: np.ndarray) -> np.ndarray:
    """
    Get the detections from the detection model.
    """
    results = detect_players_in_frames(detection_model, frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    player_detections = detections[detections.class_id == 0]
    ball_detections = detections[detections.class_id == 1]

    return player_detections, ball_detections


def init_tracking_models():
    """
    Initialize the tracking models for the player tracking system
    """
    tracker = sv.ByteTrack()
    tracker.match_thresh = 0.5
    tracker.track_buffer = 120
    ellipse_annotator = sv.EllipseAnnotator()
    traingle_annotator = sv.TriangleAnnotator()
    label_annotator = sv.LabelAnnotator()

    return tracker, ellipse_annotator, traingle_annotator, label_annotator


def train_umap_kmeans(VIDEO_PATH, detection_model, siglip_model, siglip_processor, reducer, cluster_model):
    """
    Train the UMAP and KMeans models.
    """
    # Get the video frames
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=24, end=60*24)
    
    # Get player crops
    crops = []
    for frame in tqdm(frame_generator, desc='collecting_crops'):
        player_detections, _ = get_detections(detection_model, frame)
        cropped_images = get_player_crops(frame, player_detections)
        crops += cropped_images

    # Train and get assignments
    crop_batches = create_batches(crops, 24)
    clustered_embeddings, reducer, cluster_model = assign_batch(crop_batches, siglip_model, siglip_processor, reducer, cluster_model, train=True)
    return clustered_embeddings, reducer, cluster_model


def annotation_callback(frame: np.ndarray, detection_model, siglip_model, siglip_processor, reducer, cluster_model, 
                        tracker, ellipse_annotator, traingle_annotator, label_annotator) -> np.ndarray:
    """
    A callback function to annotate the frames ellipses, labels, and team assignments.
    """

    # Get Detections
    player_detections, ball_detections = get_detections(detection_model, frame)
    if len(player_detections) == 0:
        return frame
    
    # Get player crops
    crops = get_player_crops(frame, player_detections)
    
    # Get assignments
    assignment_time = time.time()
    clustered_embeddings, reducer, cluster_model = assign_batch([crops], siglip_model, siglip_processor, reducer, cluster_model, train=False)
    assignment_time = time.time() - assignment_time
    print(f"Assignment Time: {assignment_time:.2f}s")

    # Use the assignments on the frame
    player_detections = tracker.update_with_detections(player_detections)
    player_detections.class_id = clustered_embeddings

    # Get the labels
    player_labels = [f'#{tracker_id}' for tracker_id in player_detections.tracker_id]

    # Annotate the frame
    player_annotated_frame = ellipse_annotator.annotate(frame.copy(), player_detections)
    ball_annotated_frame = traingle_annotator.annotate(player_annotated_frame, ball_detections)
    labeled_frame = label_annotator.annotate(ball_annotated_frame, detections=player_detections, labels=player_labels)

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