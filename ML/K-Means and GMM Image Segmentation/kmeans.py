import numpy as np

class KMeans:
    def __init__(self, K=3, max_iters=100):
        self.K = K#number of clusters
        self.max_iters = max_iters
    #inilizing function for kmeans
    def initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        return X[indices]
    #assigning clusters for kmeans
    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(
            X[:, np.newaxis] - centroids,
            axis=2
        )

        return np.argmin(distances, axis=1)
    #updating centroids for kmeans
    def update_centroids(self, X, labels):
        centroids = np.array([
            X[labels == k].mean(axis=0)
            for k in range(self.K)
        ])

        return centroids
    #fitting function 
    def fit(self, X):
        centroids = self.initialize_centroids(X)

        for _ in range(self.max_iters):
            labels = self.assign_clusters(X, centroids)

            new_centroids = self.update_centroids(X, labels)

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

        return labels, centroids