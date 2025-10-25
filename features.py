# features.py
import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

def compute_phash_hist(frame):
    """Compute phash and HSV histogram for a single frame."""
    phash = np.array(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).hash).astype(np.uint8).flatten()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([cv2.calcHist([hsv], [i], None, [16], [0, 256]).flatten() for i in range(3)])
    hist = hist / (hist.sum() + 1e-8)  # normalize
    return phash, hist

def process_frame(frame_idx, frame):
    phash, hist = compute_phash_hist(frame)
    return frame_idx, {'phash': phash, 'histogram': hist, 'frame': frame_idx}

def extract_features(frames_folder, video_name, max_workers=None):
    start_time = time.time()
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    frame_paths = [os.path.join(frames_folder, f) for f in frame_files]

    # Preload frames in memory
    frames = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            raise FileNotFoundError(f"Failed to read frame: {fp}")
        frames.append(img)

    features = {}
    print(f"[i] Extracting features for {len(frames)} frames using {max_workers or os.cpu_count()} workers...")

    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        futures = {executor.submit(process_frame, idx, frame): idx for idx, frame in enumerate(frames)}
        for i, future in enumerate(as_completed(futures)):
            idx, feat = future.result()
            features[idx] = feat
            if (i + 1) % 20 == 0 or (i + 1) == len(frames):
                print(f"[i] Processed {i + 1}/{len(frames)} frames")

    # Save features
    Path("features").mkdir(exist_ok=True)
    output_path = os.path.join("features", f"{video_name}_features.npy")
    np.save(output_path, features)
    end_time = time.time()
    print(f"[+] Features saved to {output_path} in {end_time - start_time:.2f}s")
    return output_path

if __name__ == "__main__":
    video_name = "jumbled_video"  # hardcoded for automation
    frames_folder = os.path.join("frames", video_name)
    extract_features(frames_folder, video_name)
