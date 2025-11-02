#!/usr/bin/env python3
"""
Optimized main pipeline for frame reordering
Features:
- Enhanced multi-feature extraction
- Beam search + 2-opt ordering
- Sliding window refinement
- Parallel processing throughout
"""

import argparse
import time
from pathlib import Path
import os
import json
import numpy as np

from frame_extractor import extract_frames
from features import extract_features
import reconstruct_order as ro

def evaluate_order_textual(pred_json_path, ground_truth_frames_folder):
    pass

def run_pipeline(video_path,
                 output_root="frames",
                 every_nth=1,
                 resize=None,
                 num_workers=None,
                 beam_width=5,
                 starts=7,
                 two_opt_iter=50,
                 max_orb_descriptors=500,
                 window_size=5,
                 swap_iter=3,
                 reverse=False,
                 num_clusters=None,
                 fps=30,
                 force_original_resolution=False):
    """
    Main pipeline with optimizations
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    
    print(f"\n{'='*60}")
    print(f"FRAME REORDERING PIPELINE - {video_name}")
    print(f"{'='*60}\n")
    
    t0 = time.time()
    
    # ===== STEP 1: Extract Frames =====
    print(f"[1/7] Extracting frames...")
    t1 = time.time()
    extracted = extract_frames(
        str(video_path),
        output_root=output_root,
        every_nth=every_nth,
        resize=resize,
        lossless=True,
        num_workers=num_workers
    )
    frames_folder = os.path.join(output_root, video_name)
    print(f"      ✓ Extracted {extracted} frames in {time.time()-t1:.2f}s\n")
    
    # ===== STEP 2: Feature Extraction =====
    print(f"[2/7] Extracting features (ORB + HSV + hashes + edges + moments)...")
    t2 = time.time()
    features_file = extract_features(
        frames_folder, video_name, 
        max_workers=num_workers, 
        force_original_resolution=force_original_resolution
    )
    print(f"      ✓ Features extracted in {time.time()-t2:.2f}s\n")
    
    # ===== STEP 3: Load Features & Compute Distances =====
    print(f"[3/7] Computing distance matrices...")
    t3 = time.time()
    
    orb, hist, phash, dhash, edges, moments, frame_paths = ro.load_features(
        features_file, video_name
    )
    
    print("      - ORB distances...")
    d_orb = ro.orb_distance_matrix_optimized(orb, max_descriptors_to_match=max_orb_descriptors)
    
    print("      - Histogram distances...")
    d_hist = ro.histogram_distance_matrix(hist)
    
    print("      - Perceptual hash distances...")
    d_phash = ro.hash_distance_matrix(phash)
    
    print("      - Difference hash distances...")
    d_dhash = ro.hash_distance_matrix(dhash)
    
    print("      - Edge histogram distances...")
    d_edge = ro.histogram_distance_matrix(edges)
    
    print("      - Color moment distances...")
    d_moment = ro.euclidean_distance_matrix(moments)
    
    print(f"      - Combining distances...")
    d_comb = ro.combine_distances(
        d_orb, d_hist, d_phash, d_dhash, d_edge, d_moment,
        orb_w=0.35, hist_w=0.15, phash_w=0.15,
        dhash_w=0.15, edge_w=0.1, moment_w=0.1
    )
    print(f"      ✓ Distance matrices computed in {time.time()-t3:.2f}s\n")
    
    # ===== STEP 4: Initial Ordering (Hierarchical or Beam Search) =====
    t4 = time.time()
    # Use hierarchical clustering if specified, or as a heuristic for very long videos
    use_hierarchical = (num_clusters is not None and num_clusters > 0) or (num_clusters is None and len(frame_paths) > 500)
    if use_hierarchical:
        num_clusters = num_clusters or int(len(frame_paths) / 20) # Apply heuristic if not specified
        print(f"[4/7] Initial ordering with hierarchical clustering (clusters={num_clusters})...")
        order = ro.hierarchical_cluster_order(d_comb, num_clusters=num_clusters)
    else:
        print(f"[4/7] Initial ordering with beam search (width={beam_width}, starts={starts})...")
        order = ro.beam_search_order(d_comb, beam_width=beam_width, starts=starts)

    print(f"      ✓ Initial order found in {time.time()-t4:.2f}s\n")
    
    # ===== STEP 5: 2-opt Refinement =====
    print(f"[5/7] 2-opt refinement (max_iter={two_opt_iter})...")
    t5 = time.time()
    order = ro.two_opt_refinement(order, d_comb, max_iter=two_opt_iter)
    print(f"      ✓ 2-opt completed in {time.time()-t5:.2f}s\n")
    
    # ===== STEP 6: Image-based Local Refinement =====
    print(f"[6/7] Local refinement with actual frames...")
    t6 = time.time()

    # Initialize shared caches for image-based refinement
    frame_cache = {}
    similarity_cache = {}
    
    print(f"      - Sliding window optimization (window={window_size})...")
    order = ro.sliding_window_refinement(order, frame_paths, window=window_size, stride=1, frame_cache=frame_cache, sim_cache=similarity_cache)

    print(f"      - Adjacent swap refinement (iter={swap_iter})...")
    order = ro.adjacent_swap_refinement(order, frame_paths, max_iter=swap_iter, frame_cache=frame_cache, sim_cache=similarity_cache)
    
    print(f"      - Final sliding window pass (window=3)...")
    order = ro.sliding_window_refinement(order, frame_paths, window=3, stride=1, frame_cache=frame_cache, sim_cache=similarity_cache)
    
    print(f"      ✓ Local refinement completed in {time.time()-t6:.2f}s\n")
    
    # ===== STEP 7: Save & Reconstruct =====
    print(f"[7/7] Saving results and reconstructing video...")
    t7 = time.time()
    
    out_json = ro.save_order_json(order, frame_paths, video_name, reverse=reverse)
    print(f"      - Order saved: {out_json}")
    
    out_vid = ro.reconstruct_video(order, frame_paths, fps=fps, reverse=reverse)
    print(f"      - Video saved: {out_vid}")
    print(f"      ✓ Output generation in {time.time()-t7:.2f}s\n")
    
    # ===== Evaluation =====
    print(f"[*] Evaluating results...")
    avg_sim = ro.evaluate_similarity(order, frame_paths)

    total_time = time.time() - t0
    
    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average frame similarity: {avg_sim:.2f}%")
    print(f"Output video: {out_vid}")
    print(f"{'='*60}\n")
    
    return {
        "pred_json": out_json,
        "reconstructed_video": out_vid,
        "avg_frame_similarity_pct": avg_sim,
        "runtime_sec": total_time,
        "frames_per_sec": extracted / total_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized frame reordering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4 --beam_width 7 --starts 10
  python main.py video.mp4 --ground_truth_frames original_frames/
  python main.py video.mp4 --resize 960 540  # Faster processing
        """
    )
    
    parser.add_argument("video_path", type=str, 
                       help="Path to jumbled input video")
    parser.add_argument("--output_root", type=str, default="frames",
                       help="Root folder for extracted frames")
    parser.add_argument("--force-original-resolution", action="store_true",
                       help="Disable dynamic downsampling in feature extraction, even if throughput is low.")

    parser.add_argument("--every_nth", type=int, default=1,
                       help="Extract every nth frame (default: 1)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                       help="Resize frames to W H for faster processing")
    parser.add_argument("--num_workers", type=int,
                       help="Number of CPU workers (default: all cores)")
    
    # Ordering parameters
    parser.add_argument("--beam_width", type=int, default=5,
                       help="Beam search width (default: 5, higher=slower but better)")
    parser.add_argument("--starts", type=int, default=7,
                       help="Number of random starts (default: 7)")
    parser.add_argument("--two_opt_iter", type=int, default=50,
                       help="2-opt max iterations (default: 50)")
    parser.add_argument("--num_clusters", type=int,
                       help="Number of clusters for hierarchical sort (default: auto). Set to 0 to disable.")
    parser.add_argument("--max_orb_descriptors", type=int, default=500,
                       help="Max ORB descriptors to use for matching (for speed, default: 500)")
    parser.add_argument("--window_size", type=int, default=5,
                       help="Sliding window size (default: 5, 3-7 recommended)")
    parser.add_argument("--swap_iter", type=int, default=3,
                       help="Adjacent swap iterations (default: 3)")
    
    # Output parameters
    parser.add_argument("--reverse", action="store_true",
                       help="Reverse the final video")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Output FPS (default: 30)")
    
    args = parser.parse_args()
    
    result = run_pipeline(
        args.video_path,
        output_root=args.output_root,
        every_nth=args.every_nth,
        resize=tuple(args.resize) if args.resize else None,
        force_original_resolution=args.force_original_resolution,
        num_workers=args.num_workers,
        beam_width=args.beam_width,
        starts=args.starts,
        two_opt_iter=args.two_opt_iter,
        num_clusters=args.num_clusters,
        max_orb_descriptors=args.max_orb_descriptors,
        window_size=args.window_size,
        swap_iter=args.swap_iter,
        reverse=args.reverse,
        fps=args.fps,
    )
    
    # Save summary
    summary_path = "output/pipeline_summary.json"
    # Convert numpy types to native python types for JSON serialization
    serializable_result = {}
    for k, v in result.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable_result[k] = v.item()
        else:
            serializable_result[k] = v

    with open(summary_path, "w") as f:
        json.dump(serializable_result, f, indent=2)
    print(f"Summary saved to: {summary_path}")