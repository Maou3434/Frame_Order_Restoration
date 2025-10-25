import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

def compute_features(frame, downsample_factor=1.0):
    """
    Enhanced feature extraction with optical flow edges and color moments
    """
    if downsample_factor > 1.0:
        new_width = int(frame.shape[1] / downsample_factor)
        new_height = int(frame.shape[0] / downsample_factor)
        # Use INTER_AREA for quality downsampling
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 1. ORB descriptors (increased features for better matching)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    
    # Feature sampling: dynamic ORB feature count based on texture variance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Heuristic mapping of texture variance to ORB feature count.
    # These thresholds and counts are tunable for speed vs. accuracy.
    if laplacian_var < 50: # Very low texture (e.g., plain sky, smooth wall)
        nfeatures_orb = 200
    elif laplacian_var < 500: # Medium texture
        nfeatures_orb = 600
    else: # High texture (e.g., detailed foliage, complex patterns)
        nfeatures_orb = 1000
    
    orb_dynamic = cv2.ORB_create(nfeatures=nfeatures_orb, scaleFactor=1.2, nlevels=8)
    kp, des = orb_dynamic.detectAndCompute(gray, None) # Use gray for ORB detection
    if des is None:
        # Ensure descriptor array is not empty, even if no features are found
        des = np.zeros((1, 32), dtype=np.uint8)
    
    # 2. Enhanced HSV histogram with finer bins
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([
        cv2.calcHist([hsv], [i], None, [32], [0, 256]).flatten() 
        for i in range(3)
    ])
    hist = hist / (hist.sum() + 1e-8)
    
    # 3. Multiple hash types for robustness
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    phash = np.array(imagehash.phash(pil_img, hash_size=8).hash).astype(np.uint8).flatten()
    dhash = np.array(imagehash.dhash(pil_img, hash_size=8).hash).astype(np.uint8).flatten()
    
    # 4. Edge density histogram (motion/structure indicator)
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256]).flatten()
    edge_hist = edge_hist / (edge_hist.sum() + 1e-8)
    
    # 5. Color moments (mean, std, skewness per channel)
    moments = []
    for i in range(3):
        channel = frame[:, :, i].flatten()
        moments.extend([
            np.mean(channel),
            np.std(channel),
            np.mean((channel - np.mean(channel))**3) / (np.std(channel)**3 + 1e-8)
        ])
    moments = np.array(moments, dtype=np.float32)
    
    return {
        'orb': des,
        'histogram': hist,
        'phash': phash,
        'dhash': dhash,
        'edges': edge_hist,
        'moments': moments
    }

def process_frame(args):
    """Process single frame (unpacked for better pickling)."""
    idx, frame_path, downsample_factor = args
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Cannot read frame: {frame_path}")
    features = compute_features(frame, downsample_factor=downsample_factor)
    return idx, features

def extract_features(frames_folder, video_name, max_workers=None):
    """
    Optimized feature extraction with better progress tracking
    """
    start_time = time.time()
    frame_files = sorted([
        f for f in os.listdir(frames_folder) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    frame_paths = [os.path.join(frames_folder, f) for f in frame_files]
    
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_folder}")
    
    # Use all cores if not specified
    if max_workers is None:
        max_workers = os.cpu_count()
    
    features = {}
    total = len(frame_paths)
    print(f"[i] Extracting features for {total} frames using {max_workers} workers...")
    
    # Process frames in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        downsample_factor = 1.0  # Start with no downsampling
        
        # --- Dynamic Resolution Logic ---
        # 1. Process an initial batch to gauge performance
        initial_batch_size = min(20, total)
        print(f"[i] Processing initial batch of {initial_batch_size} frames to determine optimal resolution...")
        start_batch_time = time.time()
        
        initial_args = [(idx, frame_paths[idx], 1.0) for idx in range(initial_batch_size)]
        initial_futures = {executor.submit(process_frame, args): args[0] for args in initial_args}
        
        for future in as_completed(initial_futures):
            idx, feat = future.result()
            features[idx] = feat
        
        elapsed_batch = time.time() - start_batch_time
        throughput = initial_batch_size / elapsed_batch if elapsed_batch > 0 else float('inf')
        print(f"[i] Initial throughput: {throughput:.2f} frames/sec.")

        # 2. Decide whether to downsample based on a throughput threshold
        if throughput < 5.0 and total > initial_batch_size: # Threshold of 5 FPS
            downsample_factor = 2.0
            print(f"[!] Throughput is low. Downsampling remaining frames by a factor of {downsample_factor} for speed.")
        
        # 3. Submit remaining tasks with the chosen downsample factor
        if total > initial_batch_size:
            remaining_args = [(idx, frame_paths[idx], downsample_factor) for idx in range(initial_batch_size, total)]
            remaining_futures = {executor.submit(process_frame, args): args[0] for args in remaining_args}
            futures.update(remaining_futures)
        
        # Collect results with progress
        completed = initial_batch_size
        for future in as_completed(futures):
            idx, feat = future.result()
            features[idx] = feat
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"[i] Processed {completed}/{total} frames ({100*completed/total:.1f}%)")
    
    # Save features
    Path("features").mkdir(exist_ok=True)
    output_path = os.path.join("features", f"{video_name}_features.npy")
    np.save(output_path, features)
    
    elapsed = time.time() - start_time
    print(f"[+] Features saved to {output_path} in {elapsed:.2f}s")
    print(f"[i] Processing speed: {total/elapsed:.1f} frames/sec")
    
    return output_path