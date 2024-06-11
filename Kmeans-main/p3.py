import os
import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_error(data, centroids, labels):
    total_error = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        total_error += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
    return total_error

def kmeans_plusplus_init(data, k):
    centroids = [data[np.random.randint(len(data))]]
    
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(data, centroid) for centroid in centroids]) for data in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        rand_value = np.random.rand()
        new_centroid_index = np.searchsorted(cumulative_probabilities, rand_value)
        centroids.append(data[new_centroid_index])
    
    return np.round(np.array(centroids), 4)

def k_means(data, k, max_iters=20):
    # Randomly initialize centroids
    centroids = kmeans_plusplus_init(data[:, :-1], k)  # Exclude the last column

    for iteration in range(1, max_iters + 1):
        # Assign each data point to the nearest centroid
        distances = np.array([np.linalg.norm(data[:, :-1] - centroid, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)

        # Update centroids based on the mean of points in each cluster
        new_centroids = np.array([data[labels == j, :-1].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(np.round(centroids, 4) == np.round(new_centroids, 4)):
            break

        centroids = np.round(new_centroids, 4)

    # Calculate error
    error = calculate_error(data[:, :-1], centroids, labels)  # Exclude the last column

    return round(error, 4)

def load_data(file_path):
    data_array_str = np.loadtxt(file_path, dtype=str)
    return np.round(data_array_str.astype(float), 4)

def run_kmeans_on_file(file_path, k_values):
    data_array = load_data(file_path)
    errors_per_k = []

    for k in k_values:
        clustering_error = k_means(data_array, k)
        print(f"For k = {k} After 20 iterations: Error = {clustering_error}")
        errors_per_k.append(clustering_error)

    # Plot the graph
    plt.plot(k_values, errors_per_k, marker='o')
    plt.title(f'Error vs. K for {os.path.basename(file_path)}')
    plt.xlabel('K (Number of Clusters)')
    plt.ylabel('Clustering Error (%)')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]

    # Check if the entered file exists
    if os.path.exists(data_file):
        # Define the range of K values
        k_values_range = list(range(2, 11))

        # Run k-means on the specified file
        run_kmeans_on_file(data_file, k_values_range)
    else:
        print(f"File '{data_file}' not found.")
        sys.exit(1)
