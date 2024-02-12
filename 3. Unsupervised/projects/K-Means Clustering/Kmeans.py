import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tolerance=0.0001, random_state=42):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.random_state = random_state

    def __find_closest_centroid(self, X, c):
        diff = np.abs(X[:, np.newaxis] - c)
        distances = np.sum(diff, axis=-1)
        idx = np.argmin(distances, axis=-1)

        return idx

    def __compute_centroid(self, X, idx):      
        c = np.zeros((self.k, X.shape[1]))

        for k in range(self.k):
            points = X[idx == k]
            c[k] = np.mean(points, axis=0)

        return c

    def fit(self, X):
        np.random.seed(self.random_state)
        randidx = np.random.permutation((X.shape[0]))
        self.centroids = X[randidx[:self.k]]

        prev_centroids = self.centroids.copy()

        for i in range(self.max_iters):
            # Assign each point to centroids
            idx = self.__find_closest_centroid(X, self.centroids)

            # Compute new centroids
            self.centroids = self.__compute_centroid(X, idx)

            # Check for convergence
            if np.linalg.norm(self.centroids - prev_centroids) < self.tolerance:
                break
            
            prev_centroids = self.centroids.copy()

        self.labels_ = idx  # Store clusters

    def predict(self, X):
        # Assign labels to new data points based on the fitted centroids
        return self.__find_closest_centroid(X, self.centroids)