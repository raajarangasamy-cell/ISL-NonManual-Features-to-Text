import numpy as np
import cv2

def preprocess_frame(frame):
    """
    Resize and normalize frame
    """
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return frame

def extract_basic_features(frames):
    """
    Converts frames into feature array
    (temporary placeholder before MediaPipe / deep model integration)
    """
    processed_frames = []

    for frame in frames:
        processed = preprocess_frame(frame)
        processed_frames.append(processed)

    return np.array(processed_frames)
