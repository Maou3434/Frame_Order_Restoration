# frame_extractor.py
"""
Extracts frames from a given video file and stores them as numbered images.
"""

import cv2
import os

def extract_frames(video_path, output_dir="frames", every_nth=1, resize=None):
    """
    Extracts frames from a video file.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save extracted frames.
        every_nth (int): Save every nth frame (default = 1 means all frames).
        resize (tuple[int, int] | None): (width, height) to resize frames. None = keep original.

    Returns:
        int: Total number of frames extracted.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_nth == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="frames", help="Output directory for frames")
    parser.add_argument("--every_nth", type=int, default=1, help="Save every nth frame")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize width height (optional)")
    args = parser.parse_args()

    total = extract_frames(
        args.video_path,
        output_dir=args.output_dir,
        every_nth=args.every_nth,
        resize=tuple(args.resize) if args.resize else None,
    )

    print(f"✅ Extracted {total} frames to {args.output_dir}/")
