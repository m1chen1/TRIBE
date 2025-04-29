import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = []
    
    def fit(self, X):
        """
        Fit DBSCAN model to the input data X.
        
        :param X: numpy array of shape (n_samples, N) where N is the length of each vector.
        """
        n_samples = X.shape[0]
        self.labels = [-1] * n_samples  # -1 means noise (unclassified points)
        visited = [False] * n_samples
        cluster_id = 0
        
        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                neighbors = self.region_query(X, i)
                
                if len(neighbors) < self.min_samples:
                    self.labels[i] = -1  # Mark as noise
                else:
                    self.expand_cluster(X, i, neighbors, cluster_id, visited)
                    cluster_id += 1

    def region_query(self, X, point_idx):
        """
        Find the neighbors of a point within the eps radius.
        
        :param X: Input data of shape (n_samples, N).
        :param point_idx: Index of the point to find neighbors for.
        
        :return: List of indices of neighbors.
        """
        neighbors = []
        for i, point in enumerate(X):
            distance = np.linalg.norm(X[point_idx] - point)
            if distance <= self.eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        """
        Expand the cluster by checking density-reachable points.
        
        :param X: Input data of shape (n_samples, N).
        :param point_idx: Index of the core point.
        :param neighbors: Indices of the neighbors of the core point.
        :param cluster_id: Current cluster ID.
        :param visited: List of visited points.
        """
        self.labels[point_idx] = cluster_id
        queue = deque(neighbors)
        
        while queue:
            current_point_idx = queue.popleft()
            
            if not visited[current_point_idx]:
                visited[current_point_idx] = True
                current_neighbors = self.region_query(X, current_point_idx)
                
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
            
            if self.labels[current_point_idx] == -1:
                self.labels[current_point_idx] = cluster_id

    def get_labels(self):
        """
        Get the cluster labels for each point.
        
        :return: List of cluster labels.
        """
        return self.labels

# Example usage:
if __name__ == "__main__":
    # Generating random data (500 samples, 16 dimensions)
    n_samples = 500
    n_features = 16
    X = np.random.random((n_samples, n_features))  # Random dataset with 16-dimensional vectors
    
    # Initialize DBSCAN with eps=0.2, min_samples=10
    dbscan = DBSCAN(eps=0.2, min_samples=10)
    
    # Fit DBSCAN to data
    dbscan.fit(X)
    
    # Get the resulting cluster labels
    labels = dbscan.get_labels()
    
    # # Plotting the clusters
    unique_labels = set(labels)
    colors = plt.get_cmap('tab20', len(unique_labels))  # Color map for multiple clusters

    plt.figure(figsize=(8, 6))
    
    # Plot points with their cluster color
    for i, label in enumerate(labels):
        color = colors(label) if label != -1 else (0.5, 0.5, 0.5)  # Grey for noise points
        plt.scatter(X[i, 0], X[i, 1], color=color, s=10)  # Plot first two features for 2D plot
    
    plt.title("DBSCAN Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
