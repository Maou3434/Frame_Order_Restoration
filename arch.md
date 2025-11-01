
# Frame Restoration Architecture

This document provides a detailed explanation of the frame restoration application's architecture, data flow, and the impact of various parameters on its performance.

## 1. High-Level Architecture

The application is a pipeline composed of several stages, each implemented in a separate Python module. The pipeline takes a jumbled video file as input and produces a reordered video with the frames in the correct sequence.

The main modules are:

-   `main.py`: The main entry point of the application that orchestrates the entire pipeline.
-   `frame_extractor.py`: Responsible for extracting frames from the input video.
-   `features.py`: Extracts a set of visual features from each frame.
-   `reconstruct_order.py`: The core module that reconstructs the correct order of frames.

## 2. Data Flow

The data flows through the pipeline in the following sequence:

```
Input Video -> [Frame Extraction] -> Frames -> [Feature Extraction] -> Features -> [Order Reconstruction] -> Reordered Video
```

1.  **Frame Extraction:** The input video is first processed by the `frame_extractor` module. It extracts individual frames from the video and saves them as image files (e.g., PNG, JPG) in a designated directory. A `meta.json` file is also created, containing metadata about the extracted frames, such as frame rate and resolution.

2.  **Feature Extraction:** The `features` module then takes the extracted frames and computes a set of visual features for each frame. These features are designed to capture the visual content of the frames, such as color, texture, and structure. The extracted features are saved in a NumPy (`.npy`) file for efficient storage and retrieval.

3.  **Order Reconstruction:** The `reconstruct_order` module is the heart of the application. It loads the extracted features and performs the following steps:
    a.  **Distance Matrix Calculation:** It computes a pairwise distance matrix between all frames based on their features. A smaller distance indicates a higher visual similarity.
    b.  **Initial Ordering:** It uses a sophisticated search algorithm (beam search or hierarchical clustering) to find a good initial order of frames based on the distance matrix.
    c.  **Refinement:** The initial order is then refined using several optimization techniques, including 2-opt refinement and image-based local refinement, to correct any remaining misplacements.
    d.  **Output Generation:** Finally, the module saves the reordered sequence of frames to a JSON file and reconstructs a new video file from the reordered frames.

## 3. Algorithm Details

The application employs a combination of computer vision and optimization algorithms to achieve its goal.

### 3.1. Feature Extraction (`features.py`)

The `compute_features` function extracts a rich set of features from each frame to capture its visual content comprehensively. The features include:

-   **ORB (Oriented FAST and Rotated BRIEF) Descriptors:** These are used to detect keypoints and compute their descriptors, which are robust to changes in scale and rotation. The number of ORB features is dynamically adjusted based on the image's texture variance to balance accuracy and speed.
-   **HSV Color Histogram:** Captures the color distribution of the frame.
-   **Perceptual and Difference Hashes (phash, dhash):** These are compact representations of the image that can be used for very fast similarity comparisons.
-   **Edge Density Histogram:** Captures the structural information of the frame by analyzing the distribution of edges.
-   **Color Moments:** Statistical measures (mean, standard deviation, and skewness) of the color distribution in each channel.

To improve performance, the feature extraction process is parallelized using `concurrent.futures.ProcessPoolExecutor`. Additionally, the application dynamically adjusts the resolution of the frames based on the initial processing throughput. If the processing is slow, it downsamples the frames to speed up the feature extraction process.

### 3.2. Order Reconstruction (`reconstruct_order.py`)

This module is responsible for finding the correct sequence of frames.

#### 3.2.1. Distance Matrix Calculation

The first step is to compute a pairwise distance matrix, where each element `(i, j)` represents the dissimilarity between frame `i` and frame `j`. The application calculates multiple distance matrices, one for each feature type, and then combines them into a single weighted distance matrix.

The distance matrix calculations are accelerated using PyTorch with GPU support (DirectML on Windows) if available, which significantly speeds up this process. If a compatible GPU is not found, the calculations fall back to NumPy on the CPU.

#### 3.2.2. Initial Ordering

Finding the optimal order of frames is a Traveling Salesperson Problem (TSP), which is NP-hard. Therefore, the application uses heuristics to find a good initial order:

-   **Beam Search (`beam_search_order`):** This is the default algorithm. It's a search algorithm that explores a graph by expanding the most promising nodes in a limited set. It is more accurate than a simple greedy search.
-   **Hierarchical Clustering (`hierarchical_cluster_order`):** For very long videos, this approach is used to reduce complexity. It first groups similar frames into clusters, then sorts the clusters, and finally sorts the frames within each cluster.

#### 3.2.3. Refinement

The initial order is then refined using the following techniques:

-   **2-opt Refinement (`two_opt_refinement`):** This is a classic TSP optimization algorithm that iteratively improves the order by reversing segments of the sequence to reduce the total distance.
-   **Image-based Local Refinement:**
    -   `adjacent_swap_refinement`: Iteratively swaps adjacent frames if the swap improves the visual similarity (measured by SSIM and NCC).
    -   `sliding_window_refinement`: Optimizes small windows of frames by finding the best permutation within that window.

These refinement steps use image-based similarity metrics, which are more accurate but slower than the feature-based distances. Caching is used to avoid redundant computations.

## 4. Parameters and Their Effects

The `main.py` script accepts several command-line arguments that allow you to control the behavior of the pipeline and trade off speed for accuracy.

### General Parameters

-   `--resize W H`: Resizes frames to the specified width and height. **Smaller resolutions lead to faster processing but may reduce accuracy.**
-   `--num_workers`: The number of CPU workers to use for parallel processing. By default, it uses all available cores.

### Ordering Parameters

-   `--beam_width`: The beam width for the beam search algorithm. **A larger beam width increases the search space and can lead to a better initial order, but it is slower.**
-   `--starts`: The number of random starting points for the beam search. **More starts increase the chance of finding a better solution but also increase the runtime.**
-   `--two_opt_iter`: The maximum number of iterations for the 2-opt refinement. **More iterations can lead to a better-refined order but will take longer.**
-   `--num_clusters`: The number of clusters for the hierarchical ordering. If not specified, it is determined automatically for long videos.
-   `--max_orb_descriptors`: The maximum number of ORB descriptors to use for matching. **A smaller number will be faster but less accurate.**
-   `--window_size`: The size of the sliding window for local refinement.
-   `--swap_iter`: The number of iterations for the adjacent swap refinement.

## 5. Performance and Bottlenecks

The performance of the pipeline depends on several factors, including the video resolution, the number of frames, and the chosen parameters. The most time-consuming parts of the pipeline are:

-   **Feature Extraction:** This is a CPU-bound task that can be slow for high-resolution videos. The dynamic resolution scaling helps to mitigate this.
-   **Distance Matrix Calculation:** This is an O(N^2) operation, where N is the number of frames. GPU acceleration is crucial for good performance here.
-   **Ordering and Refinement:** The search and optimization algorithms can also be time-consuming, especially for a large number of frames.

In summary, there is a trade-off between speed and accuracy. For a quick result, you can use a smaller resolution and smaller values for the ordering parameters. For the best possible result, you should use the original resolution and larger values for the ordering parameters, but be prepared for a longer processing time.
