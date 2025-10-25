# reconstruct_order.py
import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------
# Utility functions
# ---------------------------
def safe_mkdir(path):
    Path(path).mkdir(exist_ok=True)

def load_features(features_file):
    data = np.load(features_file, allow_pickle=True).item()
    phash_list = []
    hist_list = []
    for idx in sorted(data.keys()):
        f = data[idx]
        phash_list.append(f["phash"])
        hist_list.append(f["histogram"])
    frame_paths = [os.path.join("frames", "jumbled_video", f"frame_{i:04d}.png") for i in range(len(phash_list))]
    return np.array(phash_list, dtype=np.uint8), np.array(hist_list, dtype=np.float32), frame_paths

# ---------------------------
# Distance computations
# ---------------------------
def phash_hamming_matrix(phashes):
    N = len(phashes)
    # phashes is (N, 64) boolean-like (uint8 0/1) array.
    # Pack bits for faster XOR operations.
    # phash_uint8 will be (N, 8) where each row is 8 bytes representing 64 bits.
    phash_uint8 = np.packbits(phashes, axis=1)

    # Reshape for broadcasting:
    # phash_uint8_i (N, 1, 8)
    # phash_uint8_j (1, N, 8)
    phash_uint8_i = phash_uint8[:, np.newaxis, :]
    phash_uint8_j = phash_uint8[np.newaxis, :, :]

    # Compute XOR for all pairs: (N, N, 8)
    xor_vals_all_pairs = np.bitwise_xor(phash_uint8_i, phash_uint8_j)

    # Unpack bits for all XOR results: (N, N, 64)
    # Then sum along the bit dimension (axis=2) to get Hamming distance.
    mat = np.sum(np.unpackbits(xor_vals_all_pairs, axis=2), axis=2)
    return mat

def histogram_distance_matrix(hists):
    """
    Computes the Chi-square distance matrix for a set of histograms.
    Fully vectorized for efficiency using NumPy broadcasting.
    """
    N = hists.shape[0]
    # Reshape hists for broadcasting: hists_i (N, 1, B), hists_j (1, N, B)
    hists_i = hists[:, np.newaxis, :]
    hists_j = hists[np.newaxis, :, :]
    diff = (hists_i - hists_j) ** 2
    denom = hists_i + hists_j + 1e-10
    mat = 0.5 * np.sum(diff / denom, axis=2)
    return mat
def combine_distances(d_phash, d_hist, hist_weight=0.3):
    d = d_phash.astype(np.float32)
    d /= (d.max() + 1e-8)
    if d_hist is not None:
        h = d_hist.astype(np.float32)
        h /= (h.max() + 1e-8)
        return (1 - hist_weight) * d + hist_weight * h
    return d

# ---------------------------
# Coarse ordering
# ---------------------------
def greedy_nearest_neighbor_order(dist_matrix):
    N = dist_matrix.shape[0]
    visited = np.zeros(N, dtype=bool)
    order = [0]
    visited[0] = True
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
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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

def local_refine(order, frame_paths, max_iter=2):
    """MSE + NCC refinement, adjacent swaps, in-memory grayscale frames"""
    gray_frames = [read_gray(p) for p in frame_paths]
    for _ in range(max_iter):
        improved = False
        for i in range(len(order) - 1):
            a_idx, b_idx = order[i], order[i+1]
            a, b = gray_frames[a_idx], gray_frames[b_idx]
            mse_cur = mse_pair(a, b)
            mse_swap = mse_pair(b, a)
            if mse_swap < mse_cur:
                order[i], order[i+1] = order[i+1], order[i]
                improved = True
        if not improved:
            break
    return order

# ---------------------------
# Output
# ---------------------------
def save_order_json(order, frames):
    safe_mkdir("output")
    data = {"order_idx": order, "order_frames": [frames[i] for i in order]}
    out_path = os.path.join("output", "jumbled_video_order.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path

def reconstruct_video(order, frames, fps=30):
    safe_mkdir("output")
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    out_path = os.path.join("output", "jumbled_video_reconstructed.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for idx in tqdm(order, desc="Writing video"):
        writer.write(cv2.imread(frames[idx]))
    writer.release()
    return out_path

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    t0 = time.time()
    features_file = os.path.join("features", "jumbled_video_features.npy")
    phash, hist, frames = load_features(features_file)

    print(f"[i] Computing distance matrices...")
    d_phash = phash_hamming_matrix(phash)
    d_hist = histogram_distance_matrix(hist)
    d_comb = combine_distances(d_phash, d_hist)

    print("[i] Performing coarse ordering...")
    coarse = greedy_nearest_neighbor_order(d_comb)

    print("[i] Refining locally...")
    refined = local_refine(coarse, frames)

    out_json = save_order_json(refined, frames)
    print(f"[+] Saved predicted order to {out_json}")

    out_vid = reconstruct_video(refined, frames)
    print(f"[+] Reconstructed video written to {out_vid}")
    print(f"[i] Total runtime: {time.time() - t0:.2f}s")
