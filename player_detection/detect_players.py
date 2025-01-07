from constants import model_path, PROJECT_PATH, test_video, test_video_output
import sys
sys.path.append(PROJECT_PATH)
from utils import read_video, write_video
from ultralytics import YOLO
import cv2


def load_model(model_path):
    """Load the YOLO model"""

    model = YOLO(model_path)
    return model


def draw_bounding_boxes(image, boxes, classes, confs, class_names):
    """This function draws the bounding boxes on the image"""

    for box, cls, conf in zip(boxes, classes, confs):
        x, y, w, h = box
        x, y, w, h = int(x-(w/2)), int(y-(h/2)), int(w), int(h)
        cls = class_names[int(cls)]
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = cv2.putText(image, f"{cls} {conf:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def detect_players(video_path, output_path):
    """This function detects players in a video"""

    # Read the video
    print("Reading the video...")
    video_frames = read_video(video_path)

    # Detect players in the video
    print("Detecting players...")
    model = load_model(model_path)
    results = model(video_frames)
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


if __name__ == "__main__":
    detect_players(test_video, test_video_output)