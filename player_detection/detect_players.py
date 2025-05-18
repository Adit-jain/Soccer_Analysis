import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_detection.detection_constants import model_path, test_video, test_video_output, test_image_dir
from utils import read_video, write_video
from ultralytics import YOLO
import cv2
import random
import os
import numpy as np
import supervision as sv

##################################################### Utils ##############################################################
def load_detection_model(model_path):
    """Load the YOLO model"""

    model = YOLO(model_path)
    return model

def detect_players_in_frames(model, frames):
    """This function detects players in a frame"""

    result = model(frames)
    return result

def get_detections(detection_model, frame: np.ndarray, slice=False) -> np.ndarray:
    """
    Get the detections from the detection model.
    """

    # Define the inference callback
    def inference_callback(frame: np.ndarray) -> sv.Detections:
        """
        A callback function to convert detections to supervision format.
        """
        result = detect_players_in_frames(detection_model, frame)[0]
        return sv.Detections.from_ultralytics(result)

    # Use the slicer to get the detections if slice mode
    if slice:
        slicer = sv.InferenceSlicer(callback=inference_callback)
        detections = slicer(frame)
    # Process the frame normally if not slice mode
    else:
        detections = inference_callback(frame)

    # Get the player, ball, and referee detections
    player_detections = detections[detections.class_id == 0]
    ball_detections = detections[detections.class_id == 1]
    referee_detections = detections[detections.class_id == 2]

    # Return the separate detections
    return player_detections, ball_detections, referee_detections

def draw_bounding_boxes(image, boxes, classes, confs, class_names):
    """This function draws the bounding boxes on the image"""

    for box, cls, conf in zip(boxes, classes, confs):
        x, y, w, h = box
        x, y, w, h = int(x-(w/2)), int(y-(h/2)), int(w), int(h)
        cls = class_names[int(cls)]
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = cv2.putText(image, f"{cls} {conf:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

##################################################### Main Functions ##############################################################
def detect_players_video(video_path, output_path, model_path, frame_count=300):
    """This function detects players in a video"""

    # Read the video
    print("Reading the video...")
    video_frames = read_video(video_path, frame_count=300)

    # Detect players in the video
    print("Detecting players...")
    model = load_detection_model(model_path)
    results = detect_players_in_frames(model, video_frames)
    detected_video_frames = []

    # Draw the bounding boxes on the frames
    print("Drawing bounding boxes...")
    for index, frame in enumerate(results):
        image = video_frames[index]
        class_names = frame.names
        boxes = frame.boxes.xywh
        classes = frame.boxes.cls
        confs = frame.boxes.conf
        image = draw_bounding_boxes(image, boxes, classes, confs, class_names)
        detected_video_frames.append(image)

    # # Write the video with the players detected
    print("Writing the video...")
    write_video(detected_video_frames, output_path)


def detect_players_images(image_dir, model_path, model_2_path=None, visualize=False, samples=10):
    """This function detects players in a set of images"""

    # Load Sample number of images
    print("Loading images...")
    img_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    if samples is not None:
        img_paths = random.sample(img_paths, samples)

    # Load the model
    print("Detecting players...")
    model = load_detection_model(model_path)
    model_list = [model]

    if model_2_path is not None:
        model_2 = load_detection_model(model_2_path)
        model_list.append(model_2)

    # Predict the images
    results = [detect_players_in_frames(model_number, img_paths) for model_number in model_list]

    # Visualize and compare the images
    print("Visualizing the images...")
    if visualize:
        for image_num in range(len(results[0])):
            for model_num in results:
                model_num[image_num].show()
            input("Press Enter to continue...")

    # Return the results
    return results


def detect_players_realtime(video_path, model_path):
    """This function detects players in a video in real-time"""

    # Load the YOLO model
    model = load_detection_model(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = detect_players_in_frames(model, frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # detect_players_video(test_video, test_video_output, model_path=model_path)
    # detect_players_images(test_image_dir, model_path=model_path, visualize=True, samples=10)
    detect_players_realtime(test_video, model_path=model_path)