import argparse
import time
from pathlib import Path
from frame_extractor import extract_frames
from features import extract_features
import reconstruct_order as ro

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: extract, feature, reconstruct")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--every_nth", type=int, default=1, help="Save every nth frame")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize frames to width height")
    parser.add_argument("--reverse", action="store_true", help="Reverse the reconstructed video")
    parser.add_argument("--num_workers", type=int, help="Number of threads/processes (optional)")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    video_name = video_path.stem

    print("[i] Extracting frames...")
    total_frames = extract_frames(
        str(video_path),
        output_root="frames",
        every_nth=args.every_nth,
        resize=tuple(args.resize) if args.resize else None,
        num_workers=args.num_workers
    )

    frames_folder = Path("frames") / video_name
    print("[i] Extracting features...")
    features_file = extract_features(str(frames_folder), video_name, max_workers=args.num_workers)

    print("[i] Loading features and reconstructing video order...")
    orb, hist, phash, frame_paths = ro.load_features(features_file, video_name)

    d_orb = ro.orb_distance_matrix(orb)
    d_hist = ro.histogram_distance_matrix(hist)
    d_phash = ro.phash_hamming_matrix(phash)
    d_comb = ro.combine_distances(d_orb, d_hist, d_phash)

    coarse = ro.multi_start_greedy(d_comb, starts=3)
    refined = ro.local_refine(coarse, frame_paths)

    out_json = ro.save_order_json(refined, frame_paths, video_name, reverse=args.reverse)
    print(f"[+] Order saved to {out_json}")

    out_video = ro.reconstruct_video(refined, frame_paths, reverse=args.reverse)
    print(f"[+] Reconstructed video written to {out_video}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"[i] Total pipeline runtime: {time.time() - start_time:.2f}s")
