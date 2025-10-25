import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from numba import jit

# ---------------------------
# Utilities
# ---------------------------
def safe_mkdir(path):
    Path(path).mkdir(exist_ok=True)

def load_features(features_file, video_name):
    """Load all feature types"""
    data = np.load(features_file, allow_pickle=True).item()
    orb_list, hist_list, phash_list, dhash_list = [], [], [], []
    edge_list, moment_list = [], []
    
    for idx in sorted(data.keys()):
        f = data[idx]
        orb_list.append(f["orb"])
        hist_list.append(f["histogram"])
        phash_list.append(f["phash"])
        dhash_list.append(f["dhash"])
        edge_list.append(f["edges"])
        moment_list.append(f["moments"])
    
    frame_paths = [
        os.path.join("frames", video_name, f"frame_{i:04d}.png") 
        for i in range(len(orb_list))
    ]
    
    return (orb_list,
            np.array(hist_list, dtype=np.float32),
            np.array(phash_list, dtype=np.uint8),
            np.array(dhash_list, dtype=np.uint8),
            np.array(edge_list, dtype=np.float32),
            np.array(moment_list, dtype=np.float32),
            frame_paths)

# ---------------------------
# Optimized Distance Matrices
# ---------------------------
@jit(nopython=True)
def hamming_distance_fast(a, b):
    """Fast hamming distance using XOR"""
    return np.sum(a != b)

def orb_distance_matrix_optimized(orb_features, ratio_thresh=0.75, top_k=50):
    """
    Faster ORB matching with early termination
    """
    N = len(orb_features)
    mat = np.ones((N, N), dtype=np.float32)  # Start with max distance
    np.fill_diagonal(mat, 0)  # Zero diagonal
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    for i in range(N):
        des1 = orb_features[i]
        if des1.shape[0] < 10:  # Skip if too few features
            continue
            
        for j in range(i+1, N):
            des2 = orb_features[j]
            if des2.shape[0] < 10:
                continue
            
            try:
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if len(matches[0]) > 1 and 
                       m.distance < ratio_thresh * n.distance]
                
                # Normalized match score
                match_score = len(good) / min(len(des1), len(des2))
                distance = 1.0 - match_score
                
                mat[i, j] = mat[j, i] = distance
            except:
                pass
    
    return mat

def hash_distance_matrix(hashes):
    """Vectorized hamming distance for hash arrays"""
    N = len(hashes)
    # Expand dimensions for broadcasting
    h_i = hashes[:, np.newaxis, :]
    h_j = hashes[np.newaxis, :, :]
    # Count differing bits
    return np.sum(h_i != h_j, axis=2).astype(np.float32)

def histogram_distance_matrix(hists):
    """Chi-square distance (vectorized)"""
    hists_i = hists[:, np.newaxis, :]
    hists_j = hists[np.newaxis, :, :]
    diff = (hists_i - hists_j)**2
    denom = hists_i + hists_j + 1e-10
    return 0.5 * np.sum(diff / denom, axis=2)

def euclidean_distance_matrix(features):
    """Fast Euclidean distance using broadcasting"""
    f_i = features[:, np.newaxis, :]
    f_j = features[np.newaxis, :, :]
    return np.sqrt(np.sum((f_i - f_j)**2, axis=2))

def combine_distances(d_orb, d_hist, d_phash, d_dhash, d_edge, d_moment,
                     orb_w=0.35, hist_w=0.15, phash_w=0.15, 
                     dhash_w=0.15, edge_w=0.1, moment_w=0.1):
    """
    Weighted combination with normalization
    """
    def normalize(d):
        d_max = d.max()
        return d / d_max if d_max > 0 else d
    
    d_combined = (
        orb_w * normalize(d_orb) +
        hist_w * normalize(d_hist) +
        phash_w * normalize(d_phash) +
        dhash_w * normalize(d_dhash) +
        edge_w * normalize(d_edge) +
        moment_w * normalize(d_moment)
    )
    
    return d_combined

# ---------------------------
# Advanced Ordering Algorithms
# ---------------------------
def greedy_nearest_neighbor(dist_matrix, start_idx=0):
    """Basic greedy nearest neighbor"""
    N = dist_matrix.shape[0]
    visited = np.zeros(N, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    
    for _ in range(N-1):
        last = order[-1]
        candidates = np.where(~visited)[0]
        if len(candidates) == 0:
            break
        next_idx = candidates[np.argmin(dist_matrix[last, candidates])]
        order.append(int(next_idx))
        visited[next_idx] = True
    
    return order

def beam_search_order(dist_matrix, beam_width=3, starts=5):
    """
    Beam search for better ordering
    """
    N = dist_matrix.shape[0]
    best_order, best_cost = None, np.inf
    
    # Try multiple starting points
    start_indices = np.linspace(0, N-1, starts, dtype=int)
    
    for start_idx in start_indices:
        # Initialize beam with starting point
        beams = [([start_idx], {start_idx}, 0.0)]
        
        for step in range(N - 1):
            candidates = []
            
            for order, visited, cost in beams:
                last = order[-1]
                available = [i for i in range(N) if i not in visited]
                
                # Get top beam_width nearest neighbors
                distances = [(i, dist_matrix[last, i]) for i in available]
                distances.sort(key=lambda x: x[1])
                
                for idx, dist in distances[:beam_width]:
                    new_order = order + [idx]
                    new_visited = visited | {idx}
                    new_cost = cost + dist
                    candidates.append((new_order, new_visited, new_cost))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[2])
            beams = candidates[:beam_width]
        
        # Check best in this beam
        for order, _, cost in beams:
            if cost < best_cost:
                best_order, best_cost = order, cost
    
    return best_order

def two_opt_refinement(order, dist_matrix, max_iter=100):
    """
    2-opt local search for TSP-like improvement
    """
    N = len(order)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        for i in range(1, N - 2):
            for j in range(i + 1, N):
                if j - i == 1:
                    continue
                
                # Current cost
                current_cost = (
                    dist_matrix[order[i-1], order[i]] +
                    dist_matrix[order[j], order[(j+1) % N]]
                )
                
                # Cost after reversing segment [i:j+1]
                new_cost = (
                    dist_matrix[order[i-1], order[j]] +
                    dist_matrix[order[i], order[(j+1) % N]]
                )
                
                if new_cost < current_cost:
                    order[i:j+1] = reversed(order[i:j+1])
                    improved = True
                    break
            
            if improved:
                break
    
    return order

# ---------------------------
# Image-based Local Refinement
# ---------------------------
def read_gray_cached(paths, cache={}):
    """Read grayscale images with caching"""
    result = []
    for p in paths:
        if p not in cache:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                cache[p] = img.astype(np.float32) / 255.0
        result.append(cache.get(p))
    return result

def compute_similarity_batch(frames_a, frames_b):
    """Batch similarity computation"""
    scores = []
    for a, b in zip(frames_a, frames_b):
        if a is None or b is None:
            scores.append(0.0)
            continue
        
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        
        # Combined metric: SSIM + NCC
        # SSIM
        C1, C2 = 0.01**2, 0.03**2
        mu_a, mu_b = a.mean(), b.mean()
        sigma_a, sigma_b = a.var(), b.var()
        sigma_ab = np.mean((a - mu_a) * (b - mu_b))
        ssim = ((2*mu_a*mu_b + C1) * (2*sigma_ab + C2)) / \
               ((mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2))
        
        # NCC
        A, B = a - a.mean(), b - b.mean()
        ncc = np.sum(A * B) / (np.sqrt(np.sum(A**2) * np.sum(B**2)) + 1e-10)
        
        # Combined score (higher is better)
        score = 0.6 * ssim + 0.4 * ncc
        scores.append(score)
    
    return np.array(scores)

def sliding_window_refinement(order, frame_paths, window=5, stride=2):
    """
    Refine using sliding window optimization
    """
    cache = {}
    N = len(order)
    
    for start in range(0, N - window, stride):
        end = min(start + window, N)
        segment = order[start:end]
        segment_paths = [frame_paths[i] for i in segment]
        
        # Try all permutations of small segments (only for very small windows)
        if len(segment) <= 4:
            from itertools import permutations
            best_perm = segment
            best_score = -np.inf
            
            segment_frames = read_gray_cached(segment_paths, cache)
            
            for perm in permutations(range(len(segment))):
                perm_frames = [segment_frames[i] for i in perm]
                scores = compute_similarity_batch(
                    perm_frames[:-1], 
                    perm_frames[1:]
                )
                total_score = np.sum(scores)
                
                if total_score > best_score:
                    best_score = total_score
                    best_perm = [segment[i] for i in perm]
            
            order[start:end] = best_perm
    
    return order

def adjacent_swap_refinement(order, frame_paths, max_iter=5):
    """
    Fast adjacent swap refinement
    """
    cache = {}
    N = len(order)
    
    for iteration in range(max_iter):
        improved = False
        
        for i in range(N - 1):
            # Load 3 consecutive frames
            if i == 0:
                paths = [frame_paths[order[i]], frame_paths[order[i+1]]]
                frames = read_gray_cached(paths, cache)
                score_before = compute_similarity_batch([frames[0]], [frames[1]])[0]
            else:
                paths = [frame_paths[order[i-1]], frame_paths[order[i]], 
                        frame_paths[order[i+1]]]
                frames = read_gray_cached(paths, cache)
                score_before = (
                    compute_similarity_batch([frames[0]], [frames[1]])[0] +
                    compute_similarity_batch([frames[1]], [frames[2]])[0]
                )
            
            # Try swap
            if i + 2 < N:
                paths_after = [frame_paths[order[i-1]] if i > 0 else None,
                              frame_paths[order[i+1]], 
                              frame_paths[order[i]],
                              frame_paths[order[i+2]]]
                frames_after = [cache.get(p) if p else None for p in paths_after 
                               if p is not None]
                
                if len(frames_after) >= 3:
                    score_after = (
                        compute_similarity_batch([frames_after[0]], [frames_after[1]])[0] +
                        compute_similarity_batch([frames_after[1]], [frames_after[2]])[0]
                    )
                    
                    if score_after > score_before:
                        order[i], order[i+1] = order[i+1], order[i]
                        improved = True
        
        if not improved:
            break
    
    return order

# ---------------------------
# Output & Evaluation
# ---------------------------
def save_order_json(order, frames, video_name, reverse=False):
    safe_mkdir("output")
    if reverse:
        order = order[::-1]
    data = {
        "order_idx": order,
        "order_frames": [frames[i] for i in order]
    }
    out_path = os.path.join("output", f"{video_name}_order.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path

def reconstruct_video(order, frames, fps=30, reverse=False, codec='mp4v'):
    safe_mkdir("output")
    if reverse:
        order = order[::-1]
    
    first = cv2.imread(frames[0])
    if first is None:
        raise ValueError("Cannot read first frame")
    
    h, w = first.shape[:2]
    out_path = os.path.join("output", "reconstructed_video.mp4")
    
    # Use H264 codec if available (better compression)
    try:
        writer = cv2.VideoWriter(
            out_path, 
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264
            fps, 
            (w, h)
        )
    except:
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (w, h)
        )
    
    for idx in tqdm(order, desc="Writing video"):
        frame = cv2.imread(frames[idx])
        if frame is not None:
            writer.write(frame)
    
    writer.release()
    return out_path

def evaluate_similarity(order, frame_paths):
    """Evaluate average frame similarity"""
    cache = {}
    frames = read_gray_cached([frame_paths[i] for i in order], cache)
    
    scores = compute_similarity_batch(frames[:-1], frames[1:])
    avg_score = 100 * np.mean(scores)
    
    print(f"[i] Average frame-wise similarity: {avg_score:.2f}%")
    return avg_score