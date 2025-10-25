#!/usr/bin/env python3
"""
main.py
Orchestrates: frame extraction -> feature extraction -> ordering -> refinement -> evaluation -> reconstruct video.

Usage examples:
    python main.py path/to/jumbled_video.mp4
    python main.py path/to/jumbled_video.mp4 --reverse
    python main.py path/to/jumbled_video.mp4 --ground_truth_frames path/to/original_frames --starts 7 --max_iter 4
"""

import argparse
import time
from pathlib import Path
import os
import json

from frame_extractor import extract_frames
from features import extract_features
import reconstruct_order as ro

# ---------------------------
# Utility evaluation (textual, low tokens)
# ---------------------------
def evaluate_order_textual(pred_json_path, ground_truth_frames_folder):
    """
    Compare predicted order indices (from pred_json_path) vs
    the identity ground truth (frames in ground_truth_frames_folder named frame_0000.png ...).
    Prints a compact textual report listing sample misplacements and accuracy percentage.
    """
    with open(pred_json_path, "r") as f:
        data = json.load(f)
    pred_order = data["order_idx"]

    # Assume ground truth frames are sequentially ordered frame_0000.png ... frame_{N-1:04d}.png
    gt_files = sorted([p for p in Path(ground_truth_frames_folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    total = len(gt_files)
    if total != len(pred_order):
        print(f"[!] Warning: ground truth frame count ({total}) != predicted frame count ({len(pred_order)}). Using min.")
        total = min(total, len(pred_order))

    correct = 0
    misplacements = []
    # ground-truth index i should appear at predicted position p where pred_order[p] == i
    # But pred_order is mapping predicted_position -> original_frame_index
    # So for each predicted position i, pred_order[i] is original_index. We check if pred_order[i] == i
    for pred_pos in range(total):
        orig_idx = pred_order[pred_pos]
        if orig_idx == pred_pos:
            correct += 1
        else:
            misplacements.append((pred_pos, orig_idx))  # (predicted_position, original_index)

    accuracy = correct / total * 100.0
    print(f"--- ORDER COMPARISON (text) ---")
    print(f"Total frames compared: {total}")
    print(f"Frames exactly in correct position: {correct}")
    print(f"Misplaced frames: {len(misplacements)}")
    if misplacements:
        print("Sample misplacements (predicted_pos -> original_idx):")
        for m in misplacements[:12]:
            print(f"{m[0]} -> {m[1]}")
    print(f"Exact-position accuracy: {accuracy:.2f}%")
    return accuracy, misplacements

# ---------------------------
# Orchestration
# ---------------------------
def run_pipeline(video_path,
                 output_root="frames",
                 every_nth=1,
                 resize=None,
                 num_workers=None,
                 starts=5,
                 max_iter=3,
                 window_size=6,
                 reverse=False,
                 fps=30,
                 ground_truth_frames=None):
    video_path = Path(video_path)
    video_name = video_path.stem

    t0 = time.time()
    # 1) Extract frames
    print(f"[i] Extracting frames from {video_path} -> {output_root}/{video_name} ...")
    extracted = extract_frames(
        str(video_path),
        output_root=output_root,
        every_nth=every_nth,
        resize=resize,
        lossless=True,
        num_workers=num_workers
    )
    frames_folder = os.path.join(output_root, video_name)
    print(f"[+] Extracted {extracted} frames to {frames_folder}")

    # 2) Feature extraction
    print("[i] Extracting features (ORB + hist + phash)...")
    features_file = extract_features(frames_folder, video_name, max_workers=num_workers)
    print(f"[+] Features saved: {features_file}")

    # 3) Load features & compute distance matrices
    print("[i] Loading features and computing distance matrices...")
    orb, hist, phash, frame_paths = ro.load_features(features_file, video_name)

    print("[i] Computing ORB distance matrix...")
    d_orb = ro.orb_distance_matrix(orb)
    print("[i] Computing histogram distance matrix...")
    d_hist = ro.histogram_distance_matrix(hist)
    print("[i] Computing phash hamming matrix...")
    d_phash = ro.phash_hamming_matrix(phash)

    print("[i] Combining distances (weights: ORB=0.5, hist=0.3, phash=0.2)...")
    d_comb = ro.combine_distances(d_orb, d_hist, d_phash, orb_w=0.6, hist_w=0.2, phash_w=0.2)

    # 4) Coarse ordering (multi-start greedy)
    print(f"[i] Coarse ordering with multi-start greedy (starts={starts})...")
    coarse = ro.multi_start_greedy(d_comb, starts=starts)

    # 5) Local refinements: adjacent swaps + segment flip checks
    print(f"[i] Adjacent refinement (max_iter={max_iter})...")
    refined = ro.adjacent_refinement(coarse, frame_paths, max_iter=max_iter)

    print(f"[i] Segment flip check (window_size={window_size})...")
    refined = ro.segment_flip_check(refined, frame_paths, window_size=window_size)

    # Run another short adjacent refinement pass after flips
    refined = ro.adjacent_refinement(refined, frame_paths, max_iter=1)

    # 6) Save predicted order and reconstruct video
    out_json = ro.save_order_json(refined, frame_paths, video_name, reverse=reverse)
    print(f"[+] Predicted order saved: {out_json}")

    out_vid = ro.reconstruct_video(refined, frame_paths, fps=fps, reverse=reverse)
    print(f"[+] Reconstructed video written: {out_vid}")

    # 7) Evaluate frame-wise similarity (self-consistency)
    print("[i] Evaluating frame-wise similarity (self-consistency)...")
    avg_sim = ro.evaluate_similarity(refined, frame_paths)

    # 8) Optionally, compare against ground truth frames folder (textual)
    if ground_truth_frames:
        print("[i] Comparing predicted order to ground-truth frames folder (textual)...")
        acc, misplacements = evaluate_order_textual(out_json, ground_truth_frames)
    else:
        acc, misplacements = None, None

    total_time = time.time() - t0
    print(f"[i] Pipeline finished in {total_time:.2f}s")

    # Return key outputs
    return {
        "pred_json": out_json,
        "reconstructed_video": out_vid,
        "avg_frame_similarity_pct": avg_sim,
        "exact_position_accuracy_pct": acc,
        "misplacements_sample": misplacements,
        "runtime_sec": total_time
    }

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy-first frame reorder pipeline (10s, 30fps target)")
    parser.add_argument("video_path", type=str, help="Path to jumbled input video")
    parser.add_argument("--output_root", type=str, default="frames", help="Root folder for extracted frames")
    parser.add_argument("--every_nth", type=int, default=1, help="Save every nth frame")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize frames to WIDTH HEIGHT (optional)")
    parser.add_argument("--num_workers", type=int, help="Number of workers for feature extraction (optional)")
    parser.add_argument("--starts", type=int, default=5, help="Multi-start greedy starts")
    parser.add_argument("--max_iter", type=int, default=3, help="Adjacent refinement iterations")
    parser.add_argument("--window_size", type=int, default=6, help="Window size for segment flip check")
    parser.add_argument("--reverse", action="store_true", help="Reverse reconstructed video output")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS for reconstructed video")
    parser.add_argument("--ground_truth_frames", type=str, help="(optional) folder with original frames for textual comparison")
    args = parser.parse_args()

    out = run_pipeline(
        args.video_path,
        output_root=args.output_root,
        every_nth=args.every_nth,
        resize=tuple(args.resize) if args.resize else None,
        num_workers=args.num_workers,
        starts=args.starts,
        max_iter=args.max_iter,
        window_size=args.window_size,
        reverse=args.reverse,
        fps=args.fps,
        ground_truth_frames=args.ground_truth_frames
    )

    # print compact summary
    print("\n--- SUMMARY ---")
    print(f"Predicted order JSON: {out['pred_json']}")
    print(f"Reconstructed video: {out['reconstructed_video']}")
    print(f"Avg frame-wise similarity: {out['avg_frame_similarity_pct']:.2f}%")
    if out['exact_position_accuracy_pct'] is not None:
        print(f"Exact-position accuracy vs ground truth: {out['exact_position_accuracy_pct']:.2f}%")
    print(f"Runtime (s): {out['runtime_sec']:.2f}")
