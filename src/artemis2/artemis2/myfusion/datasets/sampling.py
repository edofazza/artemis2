import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering


def compute_optical_flow(prev_frame, curr_frame):
    """
    Compute the optical flow magnitude between two frames.

    Parameters:
        prev_frame (numpy.ndarray): Grayscale previous frame.
        curr_frame (numpy.ndarray): Grayscale current frame.

    Returns:
        float: Mean optical flow magnitude.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)


def optical_flow_sampling(folder_path, num_frames):
    """
    Sample frames from a video folder using optical flow magnitude.

    Parameters:
        folder_path (str): Path to the folder containing video frames in JPG format.
        num_frames (int): Number of frames to sample.

    Returns:
        list: List of selected frame indices.
    """
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
    if len(frame_files) < 2:
        raise ValueError("At least two frames are needed for optical flow computation.")

    prev_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)

    # Compute optical flow magnitudes in parallel
    def process_frame(i):
        curr_frame = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
        magnitude = compute_optical_flow(prev_frame, curr_frame)
        return magnitude, curr_frame

    optical_flow_magnitudes = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda i: process_frame(i), range(1, len(frame_files))), total=len(frame_files) - 1))
        for magnitude, curr_frame in results:
            optical_flow_magnitudes.append(magnitude)
            prev_frame = curr_frame

    # Normalize magnitudes and select top frames based on optical flow
    normalized_magnitudes = np.array(optical_flow_magnitudes)
    selected_indices = np.argsort(normalized_magnitudes)[-num_frames:]  # Select frames with the highest magnitudes
    selected_indices = sorted(selected_indices)  # Sort indices to preserve temporal order

    # Add the first frame as it often contains valuable context
    #selected_indices = [0] + selected_indices
    selected_indices = sorted(set(selected_indices))

    return selected_indices


def compute_histogram_difference_images(img1, img2, bins=64, grid=(4, 4)):
    """
    Computes the normalized histogram difference between two images.

    Args:
        img1: First image (numpy array).
        img2: Second image (numpy array).
        bins: Number of bins for histogram computation.
        grid: Grid size (rows, cols) for splitting the images.

    Returns:
        Normalized histogram difference value.
    """
    h, w, _ = img1.shape
    grid_h, grid_w = h // grid[0], w // grid[1]
    total_diff = 0

    for i in range(grid[0]):
        for j in range(grid[1]):
            # Define the region
            y1, y2 = i * grid_h, (i + 1) * grid_h
            x1, x2 = j * grid_w, (j + 1) * grid_w
            region1 = img1[y1:y2, x1:x2]
            region2 = img2[y1:y2, x1:x2]

            # Compute histogram differences for each color channel
            diff = 0
            for channel in range(3):  # RGB channels
                hist1 = cv2.calcHist([region1], [channel], None, [bins], [0, 256])
                hist2 = cv2.calcHist([region2], [channel], None, [bins], [0, 256])
                hist1 = hist1 / hist1.sum()  # Normalize
                hist2 = hist2 / hist2.sum()  # Normalize
                diff += np.sum(np.abs(hist1 - hist2))

            # Aggregate the differences
            total_diff += diff / 3  # Mean across channels

    # Normalize by grid dimensions and bins
    total_diff /= (grid[0] * grid[1] * bins)
    return total_diff


def select_frames_histo(image_folder: str, num_frames: int = 16, bins: int = 64, grid: tuple = (4, 4)) -> List[int]:
    """
    Selects frames from a folder of images based on histogram difference.

    Args:
        image_folder: Path to the folder containing image frames.
        num_frames: Number of frames to select.
        bins: Number of bins for histogram computation.
        grid: Grid size (rows, cols) for splitting the images.

    Returns:
        List of selected frames as numpy arrays.
    """
    # List all image files and sort them
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])

    if len(image_files) < 2:
        raise ValueError("Not enough images in the folder to compute differences.")

    frames = [cv2.imread(img_path) for img_path in image_files]
    hist_differences = []

    # Compute histogram differences between consecutive images
    for i in range(len(frames) - 1):
        hist_diff = compute_histogram_difference_images(frames[i], frames[i + 1], bins=bins, grid=grid)
        hist_differences.append((i, hist_diff))

    # Sort frames by histogram difference
    hist_differences = sorted(hist_differences, key=lambda x: x[1], reverse=True)

    # Select top N frames based on histogram difference
    selected_indices = sorted([x[0] for x in hist_differences[:num_frames]])
    return selected_indices


def select_frames_with_ssim(folder_path, num_frames):
    """
    Selects meaningful frames using SSIM from a folder containing video frames.

    Parameters:
        folder_path (str): Path to the folder containing video frames as images.
        num_frames (int): Number of frames to select.

    Returns:
        list of str: Selected frame file paths.
    """
    # Get sorted list of image file paths
    frame_paths = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])

    if len(frame_paths) == 0:
        raise ValueError("No images found in the specified folder.")

    # Load all frames in grayscale
    frames = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in frame_paths]

    # Compute SSIM dissimilarities
    dissimilarities = []
    for i in range(len(frames) - 1):
        score, _ = ssim(frames[i], frames[i + 1], full=True)
        dissimilarity = 1 - score
        dissimilarities.append(dissimilarity)

    # Rank frames based on dissimilarity
    ranked_indices = np.argsort(dissimilarities)[::-1]  # Sort in descending order of dissimilarity
    selected_indices = sorted(ranked_indices[:num_frames - 1])  # Select top M-1 indices

    # Always include the first frame
    selected_indices = [0] + selected_indices
    selected_indices = sorted(set(selected_indices))  # Remove duplicates and sort

    # Handle cases where total frames are less than required
    while len(selected_indices) < num_frames:
        selected_indices.append(len(frame_paths) - 1)

    return selected_indices


def select_frames_by_cosine_similarity(features, num_frames):
    """
    Selects meaningful frames based on cosine similarity of their features.

    Parameters:
        features (np.ndarray): Feature matrix of shape (num_images, feature_dim).
        num_frames (int): Number of frames to select.

    Returns:
        list of int: Indices of selected frames.
    """
    num_images = features.shape[0]
    if num_images <= num_frames:
        return list(range(num_images))  # If fewer frames than required, return all indices

    # Compute cosine similarity for consecutive frames
    similarities = cosine_similarity(features)
    dissimilarities = 1 - np.diag(similarities, k=1)  # Dissimilarity between consecutive frames

    # Rank by dissimilarity
    ranked_indices = np.argsort(dissimilarities)[::-1]  # Descending order
    selected_indices = sorted(ranked_indices[:num_frames - 1])  # Select top N-1 dissimilar indices

    # Always include the first frame
    selected_indices = [0] + selected_indices
    selected_indices = sorted(set(selected_indices))  # Remove duplicates and sort

    # Handle cases where total frames are less than required
    while len(selected_indices) < num_frames:
        selected_indices.append(num_images - 1)

    return selected_indices


def sample_frames_with_clustering(features, n_frames, method='kmeans', n_clusters=None):
    """
    Sample meaningful frames using clustering.

    Parameters:
        features (numpy.ndarray): Feature vectors of shape (num_frames, feature_dim).
        n_frames (int): Desired number of frames to select.
        method (str): Clustering method ('kmeans', 'hierarchical', 'spectral').
        n_clusters (int or None): Number of clusters. If None, defaults to n_frames.

    Returns:
        list: Indices of selected frames.
    """
    num_frames = features.shape[0]
    if num_frames < n_frames:
        # If the number of frames is less than the desired number, repeat the last index to reach n_frames
        indices = list(range(num_frames))
        indices += [num_frames - 1] * (n_frames - num_frames)
        return indices

    if n_clusters is None:
        n_clusters = n_frames

    # Apply clustering
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    elif method == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(features)
    elif method == 'spectral':
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42).fit(features)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # Get cluster centers or representative points
    cluster_labels = clustering.labels_
    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_center = np.mean(features[cluster_indices], axis=0)
        closest_index = cluster_indices[np.argmin(np.linalg.norm(features[cluster_indices] - cluster_center, axis=1))]
        selected_indices.append(closest_index)

    # Sort indices for consistency
    selected_indices = sorted(selected_indices)

    # Append the last index if fewer frames are selected
    while len(selected_indices) < n_frames:
        selected_indices.append(selected_indices[-1])

    return selected_indices


def sample_frames_from_folder(folder_path, n_frames, method='kmeans', n_clusters=None):
    """
    Sample meaningful frames from a folder of image features using clustering.

    Parameters:
        folder_path (str): Path to the folder containing image features as .npy files.
        n_frames (int): Desired number of frames to select.
        method (str): Clustering method ('kmeans', 'hierarchical', 'spectral').
        n_clusters (int or None): Number of clusters. If None, defaults to n_frames.

    Returns:
        list: Indices of selected frames.
    """
    # Load features from folder
    feature_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    features = np.array([np.load(os.path.join(folder_path, f)) for f in feature_files])

    #return sample_frames_with_clustering(features, n_frames, method=method, n_clusters=n_clusters)
    return select_frames_by_cosine_similarity(features, n_frames)


if __name__ == '__main__':
    folder_path = "AnimalKingdom/action_recognition/features/resnet50"
    num_frames_to_sample = 16
    rows = []
    clips = [clip for clip in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, clip))]

    def process_clip(clip):
        #indexes = optical_flow_sampling(os.path.join(folder_path, clip), num_frames_to_sample)
        #indexes = select_frames_with_ssim(os.path.join(folder_path, clip), num_frames_to_sample)
        indexes = sample_frames_from_folder(os.path.join(folder_path, clip), num_frames_to_sample)
        return {'id': clip, 'indexes': indexes}

    with ThreadPoolExecutor() as executor:
        rows = list(tqdm(executor.map(process_clip, clips), total=len(clips)))

    df = pd.DataFrame(rows)
    df.to_csv('ak_mgsampler_indexes.csv', index=False)
