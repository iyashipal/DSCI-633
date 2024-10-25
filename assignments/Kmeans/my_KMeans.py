import pandas as pd
import numpy as np

class my_KMeans:

    def __init__(self, n_clusters=8, init = "k-means++", n_init = 10, max_iter=300, tol=1e-4):
        # init = {"k-means++", "random"}
        # use euclidean distance for inertia calculation.
        # stop when either # iteration is greater than max_iter or the delta of self.inertia_ is smaller than tol.
        # repeat n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol

        self.classes_ = range(n_clusters)
        # Centroids
        self.cluster_centers_ = None
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None

    def fit(self, X):
        # X: pd.DataFrame, independent variables, float        
        # repeat self.n_init times and keep the best run 
            # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        # write your code below
        best_inertia = float('inf')  # Initialize best inertia as infinity
        best_centroids = None  # Placeholder for best centroids

        for _ in range(self.n_init):
            # Step 1: Initialize centroids
            centroids = self._init_centroids(X)

            for _ in range(self.max_iter):
                # Step 2: Assign points to the nearest centroid
                distances = self._calculate_distances(X, centroids)
                labels = np.argmin(distances, axis=1)

                # Step 3: Compute new centroids
                new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

                # Step 4: Check for convergence
                if np.all(np.abs(new_centroids - centroids) < self.tol):
                    break
                centroids = new_centroids

            # Calculate inertia for this run
            inertia = np.sum(np.min(self._calculate_distances(X, centroids), axis=1))

            # Keep the best centroids with the lowest inertia
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids

        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        distances = self._calculate_distances(X, self.cluster_centers_)
        predictions = np.argmin(distances, axis=1)
        return predictions.tolist()

    def transform(self, X):
        # Transform to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # return dists = list of [dist to centroid 1, dist to centroid 2, ...]
        # write your code below
        dists = self._calculate_distances(X, self.cluster_centers_)
        return dists.tolist()

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def _init_centroids(self, X):
        """Initialize centroids based on the selected method."""
        if self.init == "random":
            indices = np.random.choice(X.index, self.n_clusters, replace=False)
            return X.loc[indices].to_numpy()
        elif self.init == "k-means++":
            return self._kmeans_plus_plus(X)

    def _kmeans_plus_plus(self, X):
        """Initialize centroids using the k-means++ method."""
        centroids = []
        # Choose the first centroid randomly
        centroids.append(X.sample().to_numpy()[0])

        for _ in range(1, self.n_clusters):
            # Compute the squared distances to the nearest existing centroid
            dist_sq = np.min(self._calculate_distances(X, np.array(centroids))**2, axis=1)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()

            # Select the next centroid based on probabilities
            index = np.where(cumulative_probs >= r)[0][0]
            centroids.append(X.iloc[index].to_numpy())

        return np.array(centroids)

    def _calculate_distances(self, X, centroids):
        """Calculate the Euclidean distance between each point and each centroid."""
        return np.linalg.norm(X.to_numpy()[:, np.newaxis] - centroids, axis=2)





