import os
import json
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
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
            
            if des1.shape[0] > 0 and des2.shape[0] > 0:
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

def _refine_segment(segment, frame_paths, window, stride):
    """Helper function to run sliding window on a single segment."""
    # Caches must be created per-process
    frame_cache = {}
    sim_cache = {}
    
    N = len(segment)
    for start in range(0, N - window, stride):
        end = min(start + window, N)
        sub_segment = segment[start:end]
        if len(sub_segment) <= 1:
            continue

        remaining = set(sub_segment)
        best_start_node = -1
        min_avg_dist = float('inf')
        for node in sub_segment:
            avg_dist = sum(get_similarity(node, other, frame_paths, frame_cache, sim_cache) for other in sub_segment if node != other)
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_start_node = node

        new_sub_segment = [best_start_node]
        remaining.remove(best_start_node)
        while remaining:
            last = new_sub_segment[-1]
            next_node = max(remaining, key=lambda node: get_similarity(last, node, frame_paths, frame_cache, sim_cache))
            new_sub_segment.append(next_node)
            remaining.remove(next_node)
        
        segment[start:end] = new_sub_segment
    return segment

def sliding_window_refinement_parallel(order, frame_paths, window=5, stride=1, num_workers=None):
    """
    Parallel version of sliding window refinement.
    Splits the order into chunks and processes them in parallel.
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    
    chunks = np.array_split(order, num_workers)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_refine_segment, list(chunk), frame_paths, window, stride) for chunk in chunks]
        refined_chunks = [future.result() for future in as_completed(futures)]
    
    # Note: The order of results from as_completed is not guaranteed.
    # A more robust implementation would map futures back to their original chunk index.
    # For this use case, simply concatenating is sufficient as chunks are independent.
    return [item for sublist in refined_chunks for item in sublist]

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

def segment_reversal_refinement(order, frame_paths, frame_cache, sim_cache, min_len=4, max_len=15):
    """
    Checks for and reverses small segments if doing so improves boundary connections.
    This is good at fixing short sequences that are entirely backward.
    e.g., ...A-B-C-D-E... -> ...A-D-C-B-E...
    """
    print("      - Checking for reversible segments...")
    N = len(order)
    current_order = list(order)
    
    for length in range(min_len, max_len + 1):
        improved_in_pass = True
        while improved_in_pass:
            improved_in_pass = False
            for i in range(N - length + 1):
                j = i + length - 1
                
                # Boundary points
                p_before = current_order[i-1] if i > 0 else -1
                p_after = current_order[j+1] if j < N - 1 else -1
                
                # Segment endpoints
                seg_start, seg_end = current_order[i], current_order[j]

                # Score before reversal: sim(p_before, seg_start) + sim(seg_end, p_after)
                score_before = get_similarity(p_before, seg_start, frame_paths, frame_cache, sim_cache) + \
                               get_similarity(seg_end, p_after, frame_paths, frame_cache, sim_cache)

                # Score after reversal: sim(p_before, seg_end) + sim(seg_start, p_after)
                score_after = get_similarity(p_before, seg_end, frame_paths, frame_cache, sim_cache) + \
                              get_similarity(seg_start, p_after, frame_paths, frame_cache, sim_cache)
                
                if score_after > score_before:
                    current_order[i : j + 1] = reversed(current_order[i : j + 1])
                    improved_in_pass = True
    return current_order

def reinsert_misplaced_frames(order, frame_paths, frame_cache, sim_cache, max_passes=15):
    """
    Iteratively finds the single worst-placed frame ("lost" frame), removes it,
    and re-inserts it into the best possible location. This is a more stable
    "lost and found" approach than moving many frames at once.

    Args:
        order (list): The current frame order.
        frame_paths (list): List of paths to the frames.
        frame_cache (dict): Cache for loaded grayscale frames.
        sim_cache (dict): Cache for similarity scores.
        max_passes (int): Max number of frames to move.
    """
    print("      - Searching for and re-inserting 'lost' frames...")
    current_order = list(order)

    for pass_num in range(max_passes):
        N = len(current_order)
        if N < 3:
            break

        # 1. Find the weakest link in the chain
        similarities = [get_similarity(current_order[i], current_order[i+1], frame_paths, frame_cache, sim_cache) for i in range(N - 1)]
        weakest_link_idx = np.argmin(similarities)
        
        # 2. Decide which of the two frames is the "lost" one
        f1_idx, f2_idx = weakest_link_idx, weakest_link_idx + 1
        frame1, frame2 = current_order[f1_idx], current_order[f2_idx]

        # Check connection of f1 to its *other* neighbor
        sim_f1_prev = get_similarity(current_order[f1_idx-1], frame1, frame_paths, frame_cache, sim_cache) if f1_idx > 0 else similarities[weakest_link_idx]
        # Check connection of f2 to its *other* neighbor
        sim_f2_next = get_similarity(frame2, current_order[f2_idx+1], frame_paths, frame_cache, sim_cache) if f2_idx < N - 1 else similarities[weakest_link_idx]

        if sim_f1_prev < sim_f2_next:
            # frame1 is more poorly connected overall
            original_pos = f1_idx
            frame_to_move = frame1
        else:
            # frame2 is the culprit
            original_pos = f2_idx
            frame_to_move = frame2

        # 3. Remove the lost frame
        current_order.pop(original_pos)
        
        # 4. Find the best new position for it
        best_pos = -1
        best_gain = -np.inf
        
        # Calculate average similarity to use as a baseline for boundary cases
        mean_sim = np.mean(similarities)

        for i in range(len(current_order) + 1):
            prev_frame = current_order[i-1] if i > 0 else -1
            next_frame = current_order[i] if i < len(current_order) else -1

            sim_before = get_similarity(prev_frame, next_frame, frame_paths, frame_cache, sim_cache) if prev_frame != -1 and next_frame != -1 else 0
            sim_after_prev = get_similarity(prev_frame, frame_to_move, frame_paths, frame_cache, sim_cache) if prev_frame != -1 else mean_sim
            sim_after_next = get_similarity(frame_to_move, next_frame, frame_paths, frame_cache, sim_cache) if next_frame != -1 else mean_sim
            
            gain = (sim_after_prev + sim_after_next) - sim_before
            if gain > best_gain:
                best_gain = gain
                best_pos = i

        # 5. Re-insert the frame if it's a better fit and not the same spot
        if best_pos != -1 and best_pos != original_pos and best_gain > 0:
            current_order.insert(best_pos, frame_to_move)
            print(f"      - Pass {pass_num+1}: Moved frame {frame_to_move} from pos {original_pos} to {best_pos} (gain: {best_gain:.3f})")
        else:
            # Re-insert the frame at its original position if no better move was found.
            # This prevents the frame from being lost.
            current_order.insert(original_pos, frame_to_move)
            # No improvement found, stop iterating
            print(f"      - No beneficial move found for frame {frame_to_move} (best gain: {best_gain:.3f}). Halting.")
            break

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