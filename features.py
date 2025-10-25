import os
import cv2
import numpy as np
from PIL import Image
import imagehash
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_phash(image_path):
    image = Image.open(image_path)
    return np.array(imagehash.phash(image).hash).astype(np.uint8).flatten()

def compute_histogram(image_path, use_hsv=True, bins=16):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if use_hsv:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = []
    for i in range(image.shape[2]):
        channel_hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        channel_hist = cv2.normalize(channel_hist, channel_hist).flatten()
        hist.extend(channel_hist)
    return np.array(hist)

def compute_mse(image_path1, image_path2):
    img1 = cv2.imread(image_path1).astype(np.float32)
    img2 = cv2.imread(image_path2).astype(np.float32)
    return np.mean((img1 - img2) ** 2)

def compute_ncc(image_path1, image_path2):
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img1 -= img1.mean()
    img2 -= img2.mean()
    numerator = np.sum(img1 * img2)
    denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2) + 1e-8)
    return numerator / denominator

def process_single_frame(frame_path, idx, use_histogram=False, use_ncc=False):
    """Compute features for a single frame (runs in parallel)."""
    frame_features = {}
    frame_features['phash'] = compute_phash(frame_path)
    if use_histogram:
        frame_features['histogram'] = compute_histogram(frame_path)
    frame_features['mse_ref'] = frame_path
    if use_ncc:
        frame_features['ncc_ref'] = frame_path
    return idx, frame_features

def extract_features(frames_folder, video_name, use_histogram=False, use_ncc=False, max_workers=None):
    start_time = time.time()

    all_files = os.listdir(frames_folder)
    image_extensions = ('.jpg', '.jpeg', '.png')
    frame_files = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
    frame_paths = [os.path.join(frames_folder, f) for f in frame_files]

    features = {}
    total_frames = len(frame_paths)

    print(f"Starting parallel feature extraction for {total_frames} frames...")

    # Use all cores if not specified
    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        futures = {
            executor.submit(process_single_frame, frame_path, idx, use_histogram, use_ncc): idx
            for idx, frame_path in enumerate(frame_paths)
        }

        for i, future in enumerate(as_completed(futures)):
            idx, frame_features = future.result()
            features[idx] = frame_features
            if (i + 1) % 20 == 0 or (i + 1) == total_frames:
                print(f"Processed {i + 1}/{total_frames} frames")

    os.makedirs('features', exist_ok=True)
    output_path = os.path.join('features', f"{video_name}_features.npy")
    np.save(output_path, features)

    end_time = time.time()
    print(f"\n✅ Completed feature extraction in {end_time - start_time:.2f} seconds.")
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features from video frames (multi-process version)")
    parser.add_argument('--frames_root_dir', type=str, required=True, help="Path to root dir containing video frame subfolders (e.g., 'frames')")
    parser.add_argument('--video_name', type=str, required=True, help="Name of the video (subfolder in frames_root_dir)")
    parser.add_argument('--use_histogram', action='store_true', help="Include color histogram")
    parser.add_argument('--use_ncc', action='store_true', help="Include NCC for local refinement")
    parser.add_argument('--max_workers', type=int, default=None, help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    full_frames_path = os.path.join(args.frames_root_dir, args.video_name)
    extract_features(full_frames_path, args.video_name, args.use_histogram, args.use_ncc, args.max_workers)
