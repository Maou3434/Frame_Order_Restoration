import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------
def safe_mkdir(path):
    Path(path).mkdir(exist_ok=True)

def load_features(features_file, video_name):
    """
    Load ORB, histogram, and phash features from numpy file.
    Returns:
        orb: (N, descriptor_size)
        hist: (N, 48) HSV histogram
        phash: (N, 64)
        frame_paths: list of frame file paths
    """
    data = np.load(features_file, allow_pickle=True).item()
    orb_list, hist_list, phash_list = [], [], []
    for idx in sorted(data.keys()):
        f = data[idx]
        orb_list.append(f["orb"])
        hist_list.append(f["histogram"])
        phash_list.append(f["phash"])
    frame_paths = [os.path.join("frames", video_name, f"frame_{i:04d}.png") for i in range(len(orb_list))]
    return (orb_list, # Keep as a list of arrays with different lengths
            np.array(hist_list, dtype=np.float32),
            np.array(phash_list, dtype=np.uint8),
            frame_paths)

# ---------------------------
# Distance matrices
# ---------------------------
def orb_distance_matrix(orb_features, ratio_thresh=0.75):
    """
    Computes distance matrix for ORB features using a brute-force matcher.
    Distance is 1 - (normalized number of good matches).
    """
    N = len(orb_features)
    mat = np.zeros((N, N), dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    for i in range(N):
        des1 = orb_features[i]
        if des1.shape[0] == 0: continue
        for j in range(i+1, N):
            des2 = orb_features[j]
            if des2.shape[0] == 0: continue
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if len(matches[0]) > 1 and m.distance < ratio_thresh * n.distance]
            num_good = len(good_matches)
            mat[i, j] = mat[j, i] = 1.0 - (num_good / (min(len(des1), len(des2)) + 1e-8))
    return mat

def phash_hamming_matrix(phashes):
    N = len(phashes)
    phash_uint8 = np.packbits(phashes, axis=1)
    phash_uint8_i = phash_uint8[:, np.newaxis, :]
    phash_uint8_j = phash_uint8[np.newaxis, :, :]
    xor_vals = np.bitwise_xor(phash_uint8_i, phash_uint8_j)
    return np.sum(np.unpackbits(xor_vals, axis=2), axis=2)

def histogram_distance_matrix(hists):
    N = hists.shape[0]
    hists_i = hists[:, np.newaxis, :]
    hists_j = hists[np.newaxis, :, :]
    diff = (hists_i - hists_j)**2
    denom = hists_i + hists_j + 1e-10
    return 0.5*np.sum(diff/denom, axis=2)

def combine_distances(d_orb, d_hist, d_phash, orb_w=0.5, hist_w=0.3, phash_w=0.2):
    """
    Weighted combination of all three distance matrices
    """
    d = orb_w*d_orb/d_orb.max() + hist_w*d_hist/d_hist.max() + phash_w*d_phash/d_phash.max()
    return d

# ---------------------------
# Coarse ordering
# ---------------------------
def greedy_nearest_neighbor_order(dist_matrix, start_idx=0):
    N = dist_matrix.shape[0]
    visited = np.zeros(N, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    for _ in range(N-1):
        last = order[-1]
        candidates = np.where(~visited)[0]
        next_idx = candidates[np.argmin(dist_matrix[last, candidates])]
        order.append(int(next_idx))
        visited[next_idx] = True
    return order

def multi_start_greedy(dist_matrix, starts=5):
    best_order, best_score = None, np.inf
    N = dist_matrix.shape[0]
    start_indices = list(range(min(starts, N)))
    for s in start_indices:
        order = greedy_nearest_neighbor_order(dist_matrix, start_idx=s)
        score = sum(dist_matrix[order[i], order[i+1]] for i in range(len(order)-1))
        if score < best_score:
            best_order, best_score = order, score
    return best_order

# ---------------------------
# Local refinement
# ---------------------------
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)/255.0

def mse_pair(a,b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean((a-b)**2))

def ncc_pair(a,b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    A,B = a-a.mean(), b-b.mean()
    return float(np.sum(A*B)/(np.sqrt(np.sum(A**2)*np.sum(B**2))+1e-10))

def ssim_pair(a,b):
    C1,C2 = 0.01**2,0.03**2
    mu_a,mu_b = a.mean(),b.mean()
    sigma_a,sigma_b = a.var(),b.var()
    sigma_ab = np.mean((a-mu_a)*(b-mu_b))
    return ((2*mu_a*mu_b+C1)*(2*sigma_ab+C2))/((mu_a**2+mu_b**2+C1)*(sigma_a+sigma_b+C2))

def adjacent_refinement(order, frame_paths, max_iter=3):
    """
    Pairwise adjacent swap refinement using MSE/NCC/SSIM
    """
    gray_frames = [read_gray(p) for p in frame_paths]
    N = len(order)
    for _ in range(max_iter):
        improved = False
        for i in range(N-1):
            p1_idx, p2_idx = order[i], order[i+1]
            p1, p2 = gray_frames[p1_idx], gray_frames[p2_idx]
            
            # Score is a combination of errors (MSE) and dissimilarities (1-NCC, 1-SSIM)
            # Lower score is better.
            score_cur = mse_pair(p1, p2) + (1 - ncc_pair(p1, p2)) + (1 - ssim_pair(p1, p2))

            # Check if swapping with the next element is better
            if i + 2 < N:
                p3_idx = order[i+2]
                p3 = gray_frames[p3_idx]
                score_swap = mse_pair(p1, p3) + (1 - ncc_pair(p1, p3)) + (1 - ssim_pair(p1, p3))
                if score_swap < score_cur:
                    order[i+1], order[i+2] = order[i+2], order[i+1]
                    improved = True

        if not improved:
            break
    return order

def segment_flip_check(order, frame_paths, window_size=6):
    """
    Detect small reversed blocks and flip them if it improves local similarity
    """
    gray_frames = [read_gray(p) for p in frame_paths]
    N = len(order)
    for start in range(0, N - window_size + 1):
        end = start + window_size
        segment = order[start:end]
        
        def calculate_segment_score(seg):
            score = 0
            for i in range(len(seg) - 1):
                p1, p2 = gray_frames[seg[i]], gray_frames[seg[i+1]]
                score += mse_pair(p1, p2) + (1 - ncc_pair(p1, p2)) + (1 - ssim_pair(p1, p2))
            return score

        score_orig = calculate_segment_score(segment)
        score_rev = calculate_segment_score(segment[::-1])

        if score_rev < score_orig:
            order[start:end] = segment[::-1]
    return order

# ---------------------------
# Output
# ---------------------------
def save_order_json(order, frames, video_name, reverse=False):
    safe_mkdir("output")
    if reverse:
        order = order[::-1]
    data = {"order_idx": order, "order_frames":[frames[i] for i in order]}
    out_path = os.path.join("output", f"{video_name}_order.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path

def reconstruct_video(order, frames, fps=30, reverse=False):
    safe_mkdir("output")
    if reverse:
        order = order[::-1]
    first = cv2.imread(frames[0])
    h,w = first.shape[:2]
    out_path = os.path.join("output", f"reconstructed_video.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for idx in tqdm(order, desc="Writing video"):
        writer.write(cv2.imread(frames[idx]))
    writer.release()
    return out_path

# ---------------------------
# Frame-wise similarity evaluation
# ---------------------------
def evaluate_similarity(order, frame_paths):
    gray_frames = [read_gray(p) for p in frame_paths]
    scores = []
    for i in range(len(order)-1):
        a,b = gray_frames[order[i]], gray_frames[order[i+1]]
        # Similarity score 0–1
        score = 1 - mse_pair(a,b)  # simple proxy, higher is better
        scores.append(score)
    avg_score = 100*np.mean(scores)
    print(f"[i] Average frame-wise similarity: {avg_score:.2f}%")
    return avg_score
