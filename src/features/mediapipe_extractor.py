import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_face_landmarks(frame):
    """
    Extracts facial landmarks from a single frame
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    landmarks = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

    return np.array(landmarks)

def process_video_frames(frames):
    """
    Extract face features from all frames
    """
    all_features = []

    for frame in frames:
        landmarks = extract_face_landmarks(frame)
        all_features.append(landmarks)

    return all_features
