import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def davies_bouldin_index(X, labels):
    X = pd.read_csv(X)
    # Compute distance matrix
    dist_matrix = pairwise_distances(X)

    # Compute centroids
    centroids = []
    for label in np.unique(labels):
        mask = labels == label
        cluster_points = X[mask]
        centroids.append(np.mean(cluster_points, axis=0))

    # Compute pairwise distances between centroids
    centroid_dists = pairwise_distances(centroids)

    # Compute average intra-cluster distance for each cluster
    intra_cluster_dists = []
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        cluster_points = X[mask]
        centroid = centroids[i]
        if len(cluster_points) > 1:
            dists = pairwise_distances(cluster_points, [centroid])
            intra_cluster_dists.append(np.mean(dists))

    # Compute Davies-Bouldin index
    n_clusters = len(np.unique(labels))
    db_index = 0
    for i in range(n_clusters):
        max_val = -np.inf
        for j in range(n_clusters):
            if i != j:
                val = (intra_cluster_dists[i] + intra_cluster_dists[j]) / centroid_dists[i, j]
                if val > max_val:
                    max_val = val
        db_index += max_val
    return db_index / n_clusters