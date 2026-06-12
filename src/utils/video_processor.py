import cv2

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def extract_frames(video_path):
    cap = load_video(video_path)

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames
