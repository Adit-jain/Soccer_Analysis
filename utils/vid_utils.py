import cv2

def read_video(vid_path, frame_count=300):
    """This function reads a video file and yields each frame of the video"""

    frames = []
    cap = cv2.VideoCapture(vid_path)
    counter = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frames.append(frame)
        else:
            break

        counter += 1
        if counter == frame_count:
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames


def write_video(frames, out_path, fps=30):
    """This function writes the frames to a video file"""

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def show_image(image, title="Image"):
    """This function displays an image"""

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()