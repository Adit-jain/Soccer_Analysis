try:
    from tracking_constants import model_path, test_video, test_video_output, PROJECT_PATH
except ImportError:
    from player_tracking.tracking_constants import model_path, test_video, test_video_output, PROJECT_PATH

import sys
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import numpy as np
import supervision as sv
from ultralytics import YOLO
from player_detection import load_detection_model, detect_players_in_frames
from player_tracking.player_assignment import init_assignment_models, assign_players
import cv2
import time

detection_model = None
tracker = None
ellipse_annotator = None
traingle_annotator = None
label_annotator = None
assignment_model = None
processor = None
reducer = None
cluster_model = None


def annotation_callback(frame: np.ndarray, index: int) -> np.ndarray:
    """
    A callback function to annotate the frames ellipses, labels, and team assignments.
    """

    global detection_model, tracker, ellipse_annotator, traingle_annotator, label_annotator, assignment_model, processor, reducer, cluster_model

    results = detect_players_in_frames(detection_model, frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections) == 0:
        return frame
    
    player_detections = detections[detections.class_id == 0]
    ball_detections = detections[detections.class_id == 1]
    
    assignment_time = time.time()
    player_assignments = assign_players(frame, player_detections, assignment_model, processor, reducer, cluster_model)
    assignment_time = time.time() - assignment_time
    print(f"Assignment Time: {assignment_time:.2f}s")

    player_detections = tracker.update_with_detections(player_detections)
    player_detections.class_id = player_assignments
    player_labels = [f'#{tracker_id}' for tracker_id in player_detections.tracker_id]

    player_annotated_frame = ellipse_annotator.annotate(frame.copy(), player_detections)
    ball_annotated_frame = traingle_annotator.annotate(player_annotated_frame, ball_detections)
    labeled_frame = label_annotator.annotate(ball_annotated_frame, detections=player_detections, labels=player_labels)

    return labeled_frame


def init_tracking_models():
    """
    Initialize the tracking models for the player tracking system
    """
    tracker = sv.ByteTrack()
    ellipse_annotator = sv.EllipseAnnotator()
    traingle_annotator = sv.TriangleAnnotator()
    label_annotator = sv.LabelAnnotator()

    return tracker, ellipse_annotator, traingle_annotator, label_annotator


def track_players_video(video_path, output_path, model_path):
    """
    This function tracks players in a video and saves the video with the tracking annotations
    """
    global detection_model, tracker, ellipse_annotator, traingle_annotator, label_annotator, assignment_model, processor, reducer, cluster_model

    # Load the Model
    print("Initializing the model...")
    detection_model = load_detection_model(model_path)

    # Initialize the annotators
    print("Initializing the annotators...")
    tracker, ellipse_annotator, traingle_annotator, label_annotator = init_tracking_models()

    # Initialize the Player Assignment Models
    print("Initializing the player assignment models...")
    assignment_model, processor, reducer, cluster_model = init_assignment_models()

    # Process Video
    print("Processing the video...")
    sv.process_video(source_path=video_path, target_path=output_path, callback=annotation_callback)


def track_players_realtime(video_path, model_path):
    """
    This function detects players in a video in real-time
    """
    global detection_model, tracker, ellipse_annotator, traingle_annotator, label_annotator, assignment_model, processor, reducer, cluster_model

    model_init_time = time.time()
    # Load the Model
    print("Initializing the model...")
    detection_model = load_detection_model(model_path)

    # Initialize the annotators
    print("Initializing the annotators...")
    tracker, ellipse_annotator, traingle_annotator, label_annotator = init_tracking_models()

    # Initialize the Player Assignment Models
    print("Initializing the player assignment models...")
    assignment_model, processor, reducer, cluster_model = init_assignment_models()

    model_init_time = time.time() - model_init_time
    print(f"Model Initialization Time: {model_init_time:.2f}s")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            annotation_time = time.time()
            result = annotation_callback(frame, 0)
            annotation_time = time.time() - annotation_time
            print(f"Annotation Time: {annotation_time:.2f}s")
            cv2.imshow("YOLO Inference", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # track_players_video(test_video, test_video_output, model_path)
    track_players_realtime(test_video, model_path)