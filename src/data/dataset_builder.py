import os
import numpy as np
from src.pipeline.isl_pipeline import run_pipeline


DATA_PATH = "data/raw_videos"
OUTPUT_PATH = "data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def create_dataset():
    X = []
    y = []

    print("Building dataset...")

    for video_file in os.listdir(DATA_PATH):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(DATA_PATH, video_file)

        # Extract features using your pipeline
        result = run_pipeline(video_path)

        # Combine features (simple flatten strategy for now)
        face_features = result["face_features"]

        # Convert to numpy safe format
        feature_vector = np.array(face_features, dtype=object)

        X.append(feature_vector)

        # Label = filename (simple approach for now)
        label = video_file.split("_")[0]
        y.append(label)

    # Save dataset
    X = np.array(X, dtype=object)
    y = np.array(y)

    np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)

    print("Dataset created successfully!")
    print("Samples:", len(X))


if __name__ == "__main__":
    create_dataset()
