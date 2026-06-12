from src.utils.video_processor import extract_frames
from src.features.feature_extractor import extract_basic_features
from src.features.mediapipe_extractor import process_video_frames


def run_pipeline(video_path):
    print("Loading video and extracting frames...")

    # Step 1: Video → Frames
    frames = extract_frames(video_path)
    print(f"Total frames extracted: {len(frames)}")

    # Step 2: Basic preprocessing features
    basic_features = extract_basic_features(frames)
    print(f"Basic features shape: {basic_features.shape}")

    # Step 3: Non-manual features (MediaPipe)
    face_features = process_video_frames(frames)
    print(f"Face feature frames: {len(face_features)}")

    # Final output structure
    dataset = {
        "frames": frames,
        "basic_features": basic_features,
        "face_features": face_features
    }

    print("Pipeline execution completed successfully!")

    return dataset


if __name__ == "__main__":
    video_path = "data/sample.mp4"  # change later
    run_pipeline(video_path)
