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

model = None
tracker = None
ellipse_annotator = None
label_annotator = None
trace_annotator = None
smoother = None

def annotation_callback(frame: np.ndarray, index: int) -> np.ndarray:
    """A callback function to annotate the frames ellipses, labels, and traces"""

    global model, tracker, ellipse_annotator, label_annotator, trace_annotator, smoother

    results = detect_players_in_frames(model, frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    labels = [f'{tracker_id} {confidence}' for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)]

    annotated_frame = ellipse_annotator.annotate(frame.copy(), detections)
    labeled_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    traced_frame = trace_annotator.annotate(labeled_frame, detections=detections)
    return traced_frame

def track_players_video(video_path, output_path, model_path):
    """This function tracks players in a video and saves the video with the tracking annotations"""

    global model, tracker, ellipse_annotator, label_annotator, trace_annotator, smoother

    # Load the Model
    print("Initializing the model...")
    model = load_detection_model(model_path)

    # Initialize the annotators
    print("Initializing the annotators...")
    tracker = sv.ByteTrack()
    ellipse_annotator = sv.EllipseAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()
    smoother = sv.DetectionsSmoother()

    # Process Video
    print("Processing the video...")
    sv.process_video(source_path=video_path, target_path=output_path, callback=annotation_callback)


if __name__ == "__main__":
    track_players_video(test_video, test_video_output, model_path)