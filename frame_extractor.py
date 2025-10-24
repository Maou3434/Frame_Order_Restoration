# frame_extractor.py
"""
Extracts frames from a given video file and stores them as numbered images
along with a metadata file describing the extraction details.
"""

import cv2
import os
import json
from pathlib import Path

def extract_frames(video_path, output_root="frames", every_nth=1, resize=None, lossless=False):
    """
    Extracts frames from a video file.

    Args:
        video_path (str): Path to the input video.
        output_root (str): Root directory where video frames will be stored.
        every_nth (int): Save every nth frame (default=1 saves all).
        resize (tuple[int, int] | None): (width, height) to resize frames. None = keep original.
        lossless (bool): If True, saves frames as PNG instead of JPEG.

    Returns:
        int: Total number of frames extracted.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = video_path.stem  # e.g., "myvideo" from "myvideo.mp4"
    output_dir = Path(output_root) / video_name
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ext = "png" if lossless else "jpg"

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_nth == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            frame_name = f"frame_{saved_count:04d}.{ext}"
            cv2.imwrite(str(output_dir / frame_name), frame)
            saved_count += 1

        frame_idx += 1

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
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_root", type=str, default="frames", help="Root output directory")
    parser.add_argument("--every_nth", type=int, default=1, help="Save every nth frame")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize width height (optional)")
    parser.add_argument("--lossless", action="store_true", help="Save frames as PNG instead of JPEG")
    args = parser.parse_args()

    total = extract_frames(
        args.video_path,
        output_root=args.output_root,
        every_nth=args.every_nth,
        resize=tuple(args.resize) if args.resize else None,
        lossless=args.lossless,
    )

    print(f"✅ Extracted {total} frames to {Path(args.output_root) / Path(args.video_path).stem}/")
