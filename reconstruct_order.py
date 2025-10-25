import os
import json
import cv2
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import torch

# Global flag for GPU availability
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("[i] CUDA is available. Using GPU for distance computations.")
    DEVICE = torch.device("cuda")
else:
    print("[i] CUDA not available. Falling back to CPU for distance computations.")

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
# Optimized Distance Matrices (with GPU acceleration)
# ---------------------------
def orb_distance_matrix_optimized(orb_features, ratio_thresh=0.75, max_descriptors_to_match=500):
    """
    Faster ORB matching with early termination
    """
    N = len(orb_features)
    mat = np.ones((N, N), dtype=np.float32)  # Start with max distance
    np.fill_diagonal(mat, 0)  # Zero diagonal
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    for i in range(N):
        des1 = orb_features[i]
        if des1.shape[0] < 10:  # Skip if too few features to match meaningfully
            continue
        
        # Limit descriptors if specified
        if des1.shape[0] > max_descriptors_to_match:
            des1 = des1[:max_descriptors_to_match]
            
        for j in range(i+1, N):
            des2 = orb_features[j]
            if des2.shape[0] < 10: # Skip if too few features to match meaningfully
                continue
            
            try:
                if des1.shape[0] == 0 or des2.shape[0] == 0: # Ensure descriptors are not empty after slicing
                    continue
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if len(matches[0]) > 1 and 
                       m.distance < ratio_thresh * n.distance]
                
                # Normalized match score
                match_score = len(good) / min(len(des1), len(des2))
                distance = 1.0 - match_score
                
                mat[i, j] = mat[j, i] = distance
            except cv2.error: # Catch OpenCV errors, e.g., if descriptors are empty
                pass
    
    return mat

def hash_distance_matrix(hashes):
    """Vectorized hamming distance for hash arrays using PyTorch (with CPU fallback)"""
    N = len(hashes)
    if USE_CUDA:
        hashes_t = torch.from_numpy(hashes).to(DEVICE)
        # Expand dimensions for broadcasting
        h_i = hashes_t.unsqueeze(1)
        h_j = hashes_t.unsqueeze(0)
        # Count differing bits
        dist_matrix = (h_i != h_j).sum(dim=2).float()
        return dist_matrix.cpu().numpy()
    else:
        # Original NumPy implementation
        h_i = hashes[:, np.newaxis, :]
        h_j = hashes[np.newaxis, :, :]
        return np.sum(h_i != h_j, axis=2).astype(np.float32)

def histogram_distance_matrix(hists):
    """Chi-square distance (vectorized) using PyTorch (with CPU fallback)"""
    if USE_CUDA:
        hists_t = torch.from_numpy(hists).to(DEVICE)
        hists_i = hists_t.unsqueeze(1)
        hists_j = hists_t.unsqueeze(0)
        diff = (hists_i - hists_j)**2
        denom = hists_i + hists_j + 1e-10
        dist_matrix = 0.5 * torch.sum(diff / denom, dim=2)
        return dist_matrix.cpu().numpy()
    else:
        hists_i = hists[:, np.newaxis, :]
        hists_j = hists[np.newaxis, :, :]
        diff = (hists_i - hists_j)**2
        denom = hists_i + hists_j + 1e-10
        return 0.5 * np.sum(diff / denom, axis=2)

def euclidean_distance_matrix(features):
    """Fast Euclidean distance using broadcasting with PyTorch (with CPU fallback)"""
    if USE_CUDA:
        features_t = torch.from_numpy(features).to(DEVICE)
        f_i = features_t.unsqueeze(1)
        f_j = features_t.unsqueeze(0)
        dist_matrix = torch.sqrt(torch.sum((f_i - f_j)**2, dim=2))
        return dist_matrix.cpu().numpy()
    else:
        f_i = features[:, np.newaxis, :]
        f_j = features[np.newaxis, :, :]
        return np.sqrt(np.sum((f_i - f_j)**2, axis=2))

def combine_distances(d_orb, d_hist, d_phash, d_dhash, d_edge, d_moment,
                     orb_w=0.35, hist_w=0.15, phash_w=0.15, 
                     dhash_w=0.15, edge_w=0.1, moment_w=0.1):
    """
    Weighted combination with normalization
    Uses StandardScaler for more robust normalization than simple min-max scaling.
    """
    def normalize_matrix(d):
        """Scales a distance matrix using StandardScaler."""
        N = d.shape[0]
        # The diagonal is always 0, so we only scale the off-diagonal elements
        # to get a meaningful distribution of actual distances.
        off_diagonal_indices = ~np.eye(N, dtype=bool)
        distances = d[off_diagonal_indices].reshape(-1, 1)

        if distances.size == 0:
            return d # Nothing to scale

        # Scale to mean=0, std=1, then shift to be non-negative (min=0)
        scaler = StandardScaler()
        scaled_distances = scaler.fit_transform(distances)
        scaled_distances -= scaled_distances.min() # Ensure minimum is 0

        # Put the scaled values back into a new matrix
        norm_d = np.zeros_like(d, dtype=np.float32)
        norm_d[off_diagonal_indices] = scaled_distances.flatten()
        return norm_d

    d_combined = (
        orb_w * normalize_matrix(d_orb) +
        hist_w * normalize_matrix(d_hist) +
        phash_w * normalize_matrix(d_phash) +
        dhash_w * normalize_matrix(d_dhash) +
        edge_w * normalize_matrix(d_edge) +
        moment_w * normalize_matrix(d_moment)
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

def _solve_cluster_tsp(cluster_dist_matrix):
    """
    Solves the TSP problem for ordering clusters using the assignment problem solver.
    This finds the cheapest chain of connections between cluster endpoints.
    """
    row_ind, col_ind = linear_sum_assignment(cluster_dist_matrix)
    
    # Create a successor map from the assignment solution
    successors = {r: c for r, c in zip(row_ind, col_ind)}
    
    # Find the start of the path (an endpoint that is not a successor to any other)
    start_node = list(set(successors.keys()) - set(successors.values()))[0]
    
    # Reconstruct the path
    path = [start_node]
    current = start_node
    while current in successors and len(path) < len(successors):
        current = successors[current]
        path.append(current)
        
    return path

def hierarchical_cluster_order(dist_matrix, num_clusters):
    """
    Hierarchical ordering:
    1. Cluster frames.
    2. Sort frames within each cluster.
    3. Sort the clusters themselves.
    4. Chain the results.
    """
    N = dist_matrix.shape[0]

    # 1. Cluster frames using the distance matrix
    print("      - Clustering frames...")
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters, affinity='precomputed', linkage='average'
    ).fit(dist_matrix)
    
    clusters = {i: [] for i in range(num_clusters)}
    for frame_idx, cluster_id in enumerate(clustering.labels_):
        clusters[cluster_id].append(frame_idx)

    # 2. Sort frames within each cluster
    print("      - Sorting within clusters...")
    sorted_clusters = {}
    for cid, members in clusters.items():
        if not members: continue
        
        # Create a sub-matrix for the current cluster
        sub_matrix = dist_matrix[np.ix_(members, members)]
        
        # Sort using a simple greedy approach (fast for small clusters)
        start_node = np.argmin(sub_matrix.sum(axis=1)) # Start with the most 'central' frame
        local_order_indices = greedy_nearest_neighbor(sub_matrix, start_idx=start_node)
        
        # Map local indices back to global frame indices
        global_order = [members[i] for i in local_order_indices]
        sorted_clusters[cid] = global_order

    # 3. Sort the clusters
    print("      - Sorting clusters...")
    cluster_endpoints = {cid: (order[0], order[-1]) for cid, order in sorted_clusters.items()}
    cids = list(cluster_endpoints.keys())
    num_c = len(cids)
    
    # Create a distance matrix between cluster endpoints
    # Cost from cluster i to cluster j is the distance between end(i) and start(j)
    cluster_dist = np.full((num_c, num_c), np.inf)
    for i in range(num_c):
        for j in range(num_c):
            if i == j: continue
            end_i = cluster_endpoints[cids[i]][1]
            start_j = cluster_endpoints[cids[j]][0]
            cluster_dist[i, j] = dist_matrix[end_i, start_j]

    cluster_path = _solve_cluster_tsp(cluster_dist)
    ordered_cids = [cids[i] for i in cluster_path]

    # 4. Chain the results to get the final order
    final_order = [frame for cid in ordered_cids for frame in sorted_clusters[cid]]
    return final_order

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

def get_similarity(idx1, idx2, frame_paths, frame_cache, sim_cache):
    """
    Computes or retrieves from cache the similarity between two frames.
    The key for the similarity cache is a sorted tuple of indices to ensure
    sim(i, j) == sim(j, i).
    """
    key = tuple(sorted((idx1, idx2)))
    if key in sim_cache:
        return sim_cache[key]

    path1, path2 = frame_paths[idx1], frame_paths[idx2]
    frame1, frame2 = read_gray_cached([path1], frame_cache)[0], read_gray_cached([path2], frame_cache)[0]
    
    score = compute_similarity_batch([frame1], [frame2])[0]
    sim_cache[key] = score
    return score

def sliding_window_refinement(order, frame_paths, window=5, stride=2):
    """
    Refine using sliding window optimization
    """
    cache = {}
    N = len(order)
    
    # This function can be further optimized to share caches with adjacent_swap
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

def adjacent_swap_refinement(order, frame_paths, max_iter=5, frame_cache=None, sim_cache=None):
    """
    Fast adjacent swap refinement with incremental updates and caching.
    """
    if frame_cache is None: frame_cache = {}
    if sim_cache is None: sim_cache = {}
    N = len(order)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N - 1):
            # Consider swapping elements at i and i+1
            # Original segment: ... A-B-C-D ...
            # Swapped segment:  ... A-C-B-D ...
            # Indices: A=i-1, B=i, C=i+1, D=i+2
            idx_B, idx_C = order[i], order[i+1]

            # --- Calculate score before swap ---
            # We only need to evaluate the links that change: (A,B), (B,C), (C,D)
            score_before = 0
            if i > 0: # Link A-B
                score_before += get_similarity(order[i-1], idx_B, frame_paths, frame_cache, sim_cache)
            
            score_before += get_similarity(idx_B, idx_C, frame_paths, frame_cache, sim_cache) # Link B-C
            
            if i < N - 2: # Link C-D
                score_before += get_similarity(idx_C, order[i+2], frame_paths, frame_cache, sim_cache)

            # --- Calculate score after swap ---
            # New links to evaluate: (A,C), (C,B), (B,D)
            score_after = 0
            if i > 0: # Link A-C
                score_after += get_similarity(order[i-1], idx_C, frame_paths, frame_cache, sim_cache)
            
            score_after += get_similarity(idx_C, idx_B, frame_paths, frame_cache, sim_cache) # Link C-B
            
            if i < N - 2: # Link B-D
                score_after += get_similarity(idx_B, order[i+2], frame_paths, frame_cache, sim_cache)

            if score_after > score_before:
                order[i], order[i+1] = order[i+1], order[i]
                improved = True
                # A swap was made, so we restart the pass to ensure stability
                break
        
        if not improved:
            # If a full pass completes with no swaps, the order is stable
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
        # Convert numpy integers to standard python integers for JSON serialization
        "order_idx": [int(i) for i in order],
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
    avg_score = 100 * float(np.mean(scores))
    
    print(f"[i] Average frame-wise similarity: {avg_score:.2f}%")
    return avg_score