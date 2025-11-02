# Frame Restoration

This project reconstructs the correct order of frames from a jumbled video.

## 1. Source Code

### Installation

1.  **Set up Python Environment:**
    This project requires **Python 3.11.9**. It is highly recommended to use this version for optimal compatibility, especially with `torch-directml` for GPU acceleration on Windows. Newer Python versions may not have stable support for `torch-directml`.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/frame-restoration.git
    cd frame-restoration
    ```

3.  **Create a virtual environment (recommended):**
    Ensure you are using your Python 3.11.9 interpreter.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    ```bash
    pip install opencv-python numpy Pillow imagehash scikit-learn tqdm torch
    ```
    **For integrated GPU acceleration on Windows with AMD/Intel CPUs, install `torch-directml`:**
    ```bash
    pip install torch-directml
    ```

### How to Run

The main script `main.py` runs the entire pipeline.

1.  **Place your jumbled video file** in the root of the project directory. Let's assume the video is named `jumbled_video.mp4`.

2.  **Run the pipeline:**
    ```bash
    python main.py jumbled_video.mp4
    ```

    This will:
    *   Extract frames into the `frames/jumbled_video/` directory.
    *   Extract features from the frames and save them to `features/jumbled_video_features.npy`.
    *   Compute the correct frame order.
    *   Save the reordered frame sequence to `output/jumbled_video_order.json`.
    *   Reconstruct the video and save it as `output/reconstructed_video.mp4`.

### Optimal Parameters

For better accuracy, you can use the following parameters. These parameters increase the search space and iterations, which may result in longer processing time.

```bash
python main.py jumbled_video.mp4 --beam_width 7 --starts 10 --two_opt_iter 100 --window_size 5 --swap_iter 5 --force-original-resolution
```

### How to Test

To evaluate the accuracy of the reconstruction, you can provide a folder with the ground truth (original, correctly ordered) frames.

1.  **Extract the ground truth frames** into a separate directory, for example, `ground_truth_frames/`.

2.  **Run the pipeline with the `--ground_truth_frames` argument:**
    ```bash
    python main.py jumbled_video.mp4 --ground_truth_frames ground_truth_frames/
    ```

    The script will output the exact-position accuracy after the reconstruction is complete.

## 2. Algorithm Explanation

### Approach and Techniques

The frame reordering process is treated as a variation of the Traveling Salesperson Problem (TSP), where each frame is a "city" and the "distance" between them is a measure of their dissimilarity. The goal is to find the shortest path that visits every frame exactly once, which corresponds to the most logical sequence.

The algorithm uses a multi-stage approach to solve this problem:

1.  **Feature Extraction:** For each frame, a rich set of features is extracted to capture its visual content. These features include:
    *   **ORB Descriptors:** For identifying keypoints and their characteristics.
    *   **Color Histograms (HSV):** To represent the color distribution.
    *   **Perceptual Hashes (pHash, dHash):** For fast and robust image similarity comparison.
    *   **Edge Density Histograms:** To capture structural information.
    *   **Color Moments:** To represent the color distribution in a more compact way.

2.  **Distance Matrix Calculation:** A weighted distance matrix is computed between all pairs of frames. The distance is a combination of the distances calculated from each of the features mentioned above. This provides a comprehensive measure of similarity between frames.

3.  **Initial Ordering:**
    *   **Beam Search:** A beam search algorithm is used to find a good initial ordering of the frames. This is more robust than a simple greedy approach.
    *   **Hierarchical Clustering:** For very long videos, a hierarchical clustering approach is used to group similar frames together first, then sort the clusters, and finally sort the frames within each cluster.

4.  **Refinement:** The initial order is then refined using several local search techniques:
    *   **2-Opt Refinement:** A classic TSP heuristic that iteratively improves the path by reversing segments.
    *   **Sliding Window Refinement:** The order is further refined by optimizing small, overlapping windows of frames.
    *   **Adjacent Swap Refinement:**  A final pass is made to swap adjacent frames if it improves the local sequence.
    *   **Segment Reversal Refinement:**  This pass checks for and reverses small, backward sequences.
    *   **"Lost and Found" Re-insertion:**  This step identifies frames that are poorly placed in the sequence and re-inserts them in a better location.

### Design Considerations

*   **Accuracy:** The multi-feature approach and the multi-stage refinement process are designed to maximize the accuracy of the final ordering.
*   **Time Complexity:** The use of hierarchical clustering and beam search helps to manage the time complexity for large numbers of frames. The refinement steps are local and have a lower time complexity than a global search.
*   **Parallelism:** The feature extraction and some of the refinement steps are parallelized to take advantage of multi-core processors and speed up the process.
*   **GPU Acceleration:** The distance matrix calculations are accelerated using the integrated GPU (iGPU) that comes with most modern CPUs (via `torch-directml`). This provides a significant speed-up on many laptops and desktops without requiring a dedicated graphics card.

## 3. Execution Time Log

The execution time of the pipeline is logged to the console at the end of the run. A summary of the pipeline's performance is also saved to `output/pipeline_summary.json`.

Example output:
```
============================================================
PIPELINE COMPLETE
============================================================
Total runtime: 123.45s (2.06 min)
Average frame similarity: 95.21%
Exact-position accuracy: 98.50%
Output video: output/reconstructed_video.mp4
============================================================
```
