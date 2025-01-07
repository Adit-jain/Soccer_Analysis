import cv2
from ultralytics import YOLO
from constants import test_video, model_path

# Load the YOLO model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(test_video)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



