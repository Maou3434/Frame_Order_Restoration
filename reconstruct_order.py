#!/usr/bin/env python3
"""
reconstruct_order.py

Integrates with features.py (your version):
  - Loads dict-of-dict features saved by numpy.
  - Uses pHash + optional histogram for coarse ordering.
  - Optionally refines locally using MSE or NCC.

Usage examples:
  python reconstruct_order.py --video-name sample
  python reconstruct_order.py --video-name sample --hist-weight 0.2 --refine-method ncc --make-video
"""

import argparse
import json
import os
import time
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Dict, Optional

# ---------------------------
# Utility helpers
# ---------------------------

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def load_features(features_file: str) -> Dict:
    """Load dict-of-dict features saved by features.py"""
    data = np.load(features_file, allow_pickle=True).item()

    phash_list = []
    hist_list = []
    frame_paths = []

    has_hist = any("histogram" in v for v in data.values())
    for idx in sorted(data.keys()):
        f = data[idx]
        phash_bits = np.array(f["phash"], dtype=np.uint8).flatten()
        # Convert 64-bit binary vector to uint64 integer
        bits_str = "".join(str(b) for b in phash_bits)
        phash_int = int(bits_str, 2)
        phash_list.append(phash_int)
        if has_hist:
            hist_list.append(f.get("histogram", np.zeros(48)))
        # prefer mse_ref but both point to same frame
        frame_paths.append(f.get("mse_ref") or f.get("ncc_ref"))
    
    result = {
        "phash": np.array(phash_list, dtype=np.uint64),
        "hist": np.array(hist_list) if has_hist else None,
        "frames": frame_paths
    }
    return result

def list_frames_from_folder(video_name: str) -> List[str]:
    folder = os.path.join("frames", video_name)
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if not files:
        raise FileNotFoundError(f"No frames found in {folder}")
    return files

# ---------------------------
# Distance computations
# ---------------------------

def phash_hamming_matrix(phashes: np.ndarray) -> np.ndarray:
    """Compute NxN Hamming distance between uint64 phashes"""
    N = len(phashes)
    mat = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        xor_vals = np.bitwise_xor(phashes[i], phashes)
        mat[i, :] = [int(int(v).bit_count()) for v in xor_vals]
    return mat

def histogram_distance_matrix(hists: np.ndarray) -> np.ndarray:
    """Compute simple chi-square distance between histograms"""
    N = hists.shape[0]
    mat = np.zeros((N, N), dtype=np.float32)
    h = hists.astype(np.float32)
    h /= np.sum(h, axis=1, keepdims=True) + 1e-8
    for i in range(N):
        a = h[i:i+1]
        diff = (a - h) ** 2
        denom = a + h + 1e-10
        mat[i, :] = 0.5 * np.sum(diff / denom, axis=1)
    return mat

def combine_distances(d_phash: np.ndarray, d_hist: Optional[np.ndarray], hist_weight: float = 0.0):
    """Weighted sum of normalized phash and histogram distances"""
    d = d_phash.astype(np.float32)
    d /= (d.max() + 1e-8)
    if d_hist is not None and hist_weight > 0:
        h = d_hist.astype(np.float32)
        h /= (h.max() + 1e-8)
        return (1 - hist_weight) * d + hist_weight * h
    return d

# ---------------------------
# Coarse ordering
# ---------------------------

def greedy_nearest_neighbor_order(dist_matrix: np.ndarray, start_idx: int = 0) -> List[int]:
    """Greedy nearest-neighbor traversal through distance matrix"""
    N = dist_matrix.shape[0]
    visited = np.zeros(N, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    for _ in range(N - 1):
        last = order[-1]
        candidates = np.where(~visited)[0]
        next_idx = candidates[np.argmin(dist_matrix[last, candidates])]
        order.append(int(next_idx))
        visited[next_idx] = True
    return order

# ---------------------------
# Refinement
# ---------------------------

def read_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing frame: {path}")
    return img.astype(np.float32) / 255.0

def mse_pair(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean((a - b) ** 2))

def ncc_pair(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    A = a - a.mean()
    B = b - b.mean()
    denom = (np.sqrt((A**2).sum()) * np.sqrt((B**2).sum()) + 1e-10)
    return float((A * B).sum() / denom)

def local_refine(order: List[int], frame_paths: List[str], method="mse", max_iter=2) -> List[int]:
    """Simple local adjacent-swap refinement"""
    assert method in ("mse", "ncc", "none")
    if method == "none":
        return order
    for _ in range(max_iter):
        improved = False
        for i in range(len(order) - 1):
            a_idx, b_idx = order[i], order[i+1]
            img_a = read_gray(frame_paths[a_idx])
            img_b = read_gray(frame_paths[b_idx])
            score_cur = mse_pair(img_a, img_b) if method == "mse" else ncc_pair(img_a, img_b)
            score_swapped = mse_pair(img_b, img_a) if method == "mse" else ncc_pair(img_b, img_a)
            better = score_swapped < score_cur if method == "mse" else score_swapped > score_cur
            if better:
                order[i], order[i+1] = order[i+1], order[i]
                improved = True
        if not improved:
            break
    return order

# ---------------------------
# Output helpers
# ---------------------------

def save_order_json(order, frames, video_name):
    safe_mkdir("output")
    data = {"order_idx": order, "order_frames": [frames[i] for i in order]}
    out_path = os.path.join("output", f"{video_name}_order.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path

def reconstruct_video(order, frames, video_name, fps=30):
    safe_mkdir("output")
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    out_path = os.path.join("output", f"{video_name}_reconstructed.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for idx in tqdm(order, desc="Writing video"):
        frame = cv2.imread(frames[idx])
        writer.write(frame)
    writer.release()
    return out_path

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Reconstruct frame order from precomputed features")
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--features-file", default=None)
    parser.add_argument("--hist-weight", type=float, default=0.0)
    parser.add_argument("--refine-method", choices=["mse", "ncc", "none"], default="mse")
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    features_file = args.features_file or os.path.join("features", f"{args.video_name}_features.npy")
    print(f"[i] Loading features: {features_file}")
    t0 = time.time()
    feats = load_features(features_file)
    phash, hist, frames = feats["phash"], feats["hist"], feats["frames"]

    print(f"[i] {len(frames)} frames loaded.")
    d_phash = phash_hamming_matrix(phash)
    d_hist = histogram_distance_matrix(hist) if hist is not None and args.hist_weight > 0 else None
    d_comb = combine_distances(d_phash, d_hist, args.hist_weight)

    print("[i] Performing coarse ordering...")
    coarse = greedy_nearest_neighbor_order(d_comb)

    print(f"[i] Refining locally using {args.refine_method}...")
    refined = local_refine(coarse, frames, method=args.refine_method)

    out_json = save_order_json(refined, frames, args.video_name)
    print(f"[+] Saved predicted order to {out_json}")
    print(f"[i] Runtime: {time.time() - t0:.2f}s")

    if args.make_video:
        out_vid = reconstruct_video(refined, frames, args.video_name, fps=args.fps)
        print(f"[+] Reconstructed video written to {out_vid}")

if __name__ == "__main__":
    main()
