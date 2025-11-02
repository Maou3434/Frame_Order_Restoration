# frame_extractor.py
"""
Extracts frames from a video file and stores them as numbered images
along with a metadata file describing the extraction details.
Multithreaded by default for faster processing.
"""

import cv2
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time

def save_frame(frame_idx, frame, output_dir, resize, ext):
    """
    Saves a single frame to disk.

    Args:
        frame_idx (int): The index of the frame to be saved.
        frame (np.ndarray): The frame image data.
        output_dir (Path): The directory to save the frame in.
        resize (tuple[int, int] | None): Target (W, H) to resize the frame.
        ext (str): The file extension to use (e.g., 'png' or 'jpg').

    Returns:
        int: The index of the saved frame.
    """
    if resize:
        frame = cv2.resize(frame, resize)
    frame_name = f"frame_{frame_idx:04d}.{ext}"
    cv2.imwrite(str(output_dir / frame_name), frame)
    return frame_idx

def extract_frames(video_path, output_root="frames", every_nth=1, resize=None, lossless=True, num_workers=None):
    """
    Extract frames from a video using a multithreaded approach.

    Args:
        video_path (str): Path to the input video file.
        output_root (str): Root directory where frames will be saved.
        every_nth (int): Interval for frame extraction (e.g., 1 for every frame).
        resize (tuple[int, int] | None): Optional (width, height) to resize frames.
        lossless (bool): If True, save frames as PNG; otherwise, save as JPG.
        num_workers (int | None): Number of threads. Defaults to min(8, CPU cores).

    Returns:
        int: Number of frames extracted.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = video_path.stem
    output_dir = Path(output_root) / video_name
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ext = "png" if lossless else "jpg"

    if num_workers is None:
        num_workers = min(8, multiprocessing.cpu_count())

    frame_idx = 0
    saved_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_nth == 0:
                futures.append(executor.submit(save_frame, saved_count, frame, output_dir, resize, ext))
                saved_count += 1

            frame_idx += 1

        # Wait for all frames to finish
        for f in as_completed(futures):
            f.result()

    cap.release()

    # Save metadata
    metadata = {
        "video_name": video_path.name,
        "frame_count": saved_count,
        "fps": fps,
        "resolution": [width, height],
        "resize": resize if resize else [width, height],
        "format": ext
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return saved_count


if __name__ == "__main__":
    start_time = time.time()
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a video file (multithreaded).")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_root", type=str, default="frames", help="Root output directory")
    parser.add_argument("--every_nth", type=int, default=1, help="Save every nth frame")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize width height (optional)")
    parser.add_argument("--num_workers", type=int, help="Number of threads (optional)")
    args = parser.parse_args()

    total = extract_frames(
        args.video_path,
        output_root=args.output_root,
        every_nth=args.every_nth,
        resize=tuple(args.resize) if args.resize else None,
        num_workers=args.num_workers
    )

    print(f"✅ Extracted {total} frames to {Path(args.output_root) / Path(args.video_path).stem}/")
    end_time = time.time()
    print(f"Script ran for {end_time - start_time:.2f} seconds.")
