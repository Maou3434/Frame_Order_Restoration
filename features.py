import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

def compute_features(frame):
    """
    Enhanced feature extraction with optical flow edges and color moments
    """
    # 1. ORB descriptors (increased features for better matching)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    kp, des = orb.detectAndCompute(frame, None)
    if des is None:
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    """Process single frame (unpacked for better pickling)"""
    idx, frame_path = args
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Cannot read frame: {frame_path}")
    features = compute_features(frame)
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
    print(f"[i] Extracting enhanced features for {total} frames using {max_workers} workers...")
    
    # Process frames in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        args_list = [(idx, fp) for idx, fp in enumerate(frame_paths)]
        futures = {executor.submit(process_frame, args): args[0] for args in args_list}
        
        # Collect results with progress
        completed = 0
        for future in as_completed(futures):
            idx, feat = future.result()
            features[idx] = feat
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"[i] Processed {completed}/{total} frames ({100*completed/total:.1f}%)")
    
    # Save features
    Path("features").mkdir(exist_ok=True)
    output_path = os.path.join("features", f"{video_name}_features.npy")
    np.save(output_path, features)
    
    elapsed = time.time() - start_time
    print(f"[+] Features saved to {output_path} in {elapsed:.2f}s")
    print(f"[i] Processing speed: {total/elapsed:.1f} frames/sec")
    
    return output_path