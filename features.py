import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

# -------------------------------
# Feature extraction functions
# -------------------------------
def compute_orb_hist_phash(frame):
    # 1. ORB descriptors
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(frame, None)
    if des is None:
        des = np.zeros((1, 32), dtype=np.uint8)
    des_flat = des.flatten()[:1024]  # limit to 1024 elements
    des_flat = np.pad(des_flat, (0, max(0, 1024 - des_flat.size)), 'constant')

    # 2. HSV histogram
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([cv2.calcHist([hsv], [i], None, [16], [0, 256]).flatten() for i in range(3)])
    hist = hist / (hist.sum() + 1e-8)

    # 3. phash
    phash = np.array(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).hash).astype(np.uint8).flatten()

    return des_flat, hist, phash

def process_frame(idx, frame):
    des, hist, phash = compute_orb_hist_phash(frame)
    return idx, {'orb': des, 'histogram': hist, 'phash': phash}

# -------------------------------
# Main extraction function
# -------------------------------
def extract_features(frames_folder, video_name, max_workers=None):
    start_time = time.time()
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    frame_paths = [os.path.join(frames_folder, f) for f in frame_files]

    # Preload frames
    frames = [cv2.imread(fp) for fp in frame_paths]

    features = {}
    print(f"[i] Extracting ORB + HSV + phash features for {len(frames)} frames...")
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
