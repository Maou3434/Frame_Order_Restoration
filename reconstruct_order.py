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

# --- Device Selection for Acceleration ---
# We prioritize using the integrated GPU via DirectML, falling back to CPU.
DEVICE = None
USE_GPU = False

try:
    import torch_directml
    if torch_directml.is_available():
        DEVICE = torch_directml.device()
        USE_GPU = True
        print("[i] DirectML device found. Using integrated GPU for acceleration.")
except (ImportError, RuntimeError, AttributeError):
    # This block will be hit if torch_directml is not installed or no compatible GPU is found
    DEVICE = torch.device("cpu")
    USE_GPU = False
    print("[!] Warning: DirectML not available or could not be initialized.")
    print("    ensure you have a compatible Python version (e.g., 3.11) and run: pip install torch-directml")

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
                # k=2 for ratio test. Guard against cases with < 2 matches.
                matches = bf.knnMatch(des1, des2, k=2)
                # Apply ratio test as per Lowe's paper
                # Filter matches to ensure there are two neighbors, then apply the ratio test.
                good = [m for m, n in (match for match in matches if len(match) == 2) if m.distance < ratio_thresh * n.distance]
                
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
    if USE_GPU:
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
    if USE_GPU:
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
    if USE_GPU:
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

def sliding_window_refinement(order, frame_paths, window=5, stride=2, frame_cache=None, sim_cache=None):
    """
    Refine using sliding window optimization with shared caches.
    """
    if frame_cache is None: frame_cache = {}
    if sim_cache is None: sim_cache = {}
    N = len(order)
    
    for start in range(0, N - window, stride):
        end = min(start + window, N)
        segment = order[start:end]
        if len(segment) <= 1:
            continue
        
        # Perform a greedy search within the window to find a better local order
        # This is much faster than checking all permutations for windows > 4
        remaining = set(segment)
        
        # Find the best starting node within the segment
        best_start_node = -1
        min_avg_dist = float('inf')
        for node in segment:
            avg_dist = sum(get_similarity(node, other, frame_paths, frame_cache, sim_cache) for other in segment if node != other)
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_start_node = node

        new_segment = [best_start_node]
        remaining.remove(best_start_node)
        while remaining:
            last = new_segment[-1]
            next_node = max(remaining, key=lambda node: get_similarity(last, node, frame_paths, frame_cache, sim_cache))
            new_segment.append(next_node)
            remaining.remove(next_node)
        
        order[start:end] = new_segment
    
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
            idx_A = order[i-1] if i > 0 else -1
            idx_B, idx_C = order[i], order[i+1]
            idx_D = order[i+2] if i < N - 2 else -1

            # --- Calculate score before swap ---
            # We evaluate the total similarity of the local chain.
            # Original links: (A,B) and (B,C) and (C,D)
            sim_AB = get_similarity(idx_A, idx_B, frame_paths, frame_cache, sim_cache) if idx_A != -1 else 0
            sim_BC = get_similarity(idx_B, idx_C, frame_paths, frame_cache, sim_cache)
            sim_CD = get_similarity(idx_C, idx_D, frame_paths, frame_cache, sim_cache) if idx_D != -1 else 0
            score_before = sim_AB + sim_BC + sim_CD

            # --- Calculate score after swap ---
            # New links: (A,C) and (C,B) and (B,D)
            sim_AC = get_similarity(idx_A, idx_C, frame_paths, frame_cache, sim_cache) if idx_A != -1 else 0
            sim_CB = get_similarity(idx_C, idx_B, frame_paths, frame_cache, sim_cache) # Same as sim_BC
            sim_BD = get_similarity(idx_B, idx_D, frame_paths, frame_cache, sim_cache) if idx_D != -1 else 0
            score_after = sim_AC + sim_CB + sim_BD

            if score_after > score_before:
                order[i], order[i+1] = order[i+1], order[i]
                improved = True
                # A swap was made. We could restart the pass, but continuing
                # is often faster and sufficient, especially if we iterate.
                # For stability, we'll break and restart the pass.
                break 
        
        if not improved:
            # If a full pass completes with no swaps, the order is stable
            break
    
    return order

def reinsert_misplaced_frames(order, frame_paths, frame_cache, sim_cache, similarity_threshold=0.5, search_step=5):
    """
    Finds poorly placed frames and attempts to re-insert them into a better location.
    This is a "lost and found" pass for frames that local refinements can't fix.

    Args:
        order (list): The current frame order.
        frame_paths (list): List of paths to the frames.
        frame_cache (dict): Cache for loaded grayscale frames.
        sim_cache (dict): Cache for similarity scores.
        similarity_threshold (float): A frame is considered "lost" if its similarity to
                                     both neighbors is below this value.
        search_step (int): How many positions to skip when searching for a new home.
                           A higher value is faster but less exhaustive.
    """
    print("      - Searching for and re-inserting 'lost' frames...")
    N = len(order)
    lost_frames = []

    # 1. Identify "lost" frames based on low similarity to neighbors
    for i in range(N):
        prev_idx = order[i-1] if i > 0 else -1
        curr_idx = order[i]
        next_idx = order[i+1] if i < N - 1 else -1

        sim_to_prev = get_similarity(prev_idx, curr_idx, frame_paths, frame_cache, sim_cache) if prev_idx != -1 else 1.0
        sim_to_next = get_similarity(curr_idx, next_idx, frame_paths, frame_cache, sim_cache) if next_idx != -1 else 1.0

        if sim_to_prev < similarity_threshold and sim_to_next < similarity_threshold:
            lost_frames.append((i, curr_idx))

    if not lost_frames:
        print("      - No lost frames found. Skipping.")
        return order

    print(f"      - Found {len(lost_frames)} potential lost frames. Attempting re-insertion.")
    
    current_order = list(order)
    # Process from last to first to not mess up indices of earlier items
    for (original_pos, frame_to_move) in reversed(lost_frames):
        # Temporarily remove the frame
        current_order.pop(original_pos)
        
        best_pos = -1
        best_gain = -np.inf

        # 2. Find the best place to re-insert it
        # The cost is the similarity gain from inserting frame_to_move between two other frames.
        for i in range(0, len(current_order) + 1, search_step):
            prev_frame = current_order[i-1] if i > 0 else -1
            next_frame = current_order[i] if i < len(current_order) else -1

            sim_before = get_similarity(prev_frame, next_frame, frame_paths, frame_cache, sim_cache) if prev_frame != -1 and next_frame != -1 else 0
            sim_after = get_similarity(prev_frame, frame_to_move, frame_paths, frame_cache, sim_cache) + \
                        get_similarity(frame_to_move, next_frame, frame_paths, frame_cache, sim_cache)
            
            if sim_after - sim_before > best_gain:
                best_gain = sim_after - sim_before
                best_pos = i

        # 3. Insert it into the best found position
        if best_pos != -1:
            current_order.insert(best_pos, frame_to_move)

    return current_order
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