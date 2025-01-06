import cv2
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

vid_path = r"D:\Datasets\SoccerNet\Data\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley\1_224p.mkv"
model = YOLO(r"Models/Trained/yolov11/Third/weights/best.pt")

cap = cv2.VideoCapture(vid_path)
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

cap.release()
cv2.destroyAllWindows()



