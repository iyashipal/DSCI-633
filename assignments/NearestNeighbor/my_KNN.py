import pandas as pd
import numpy as np
from collections import Counter

#DID NOT USE HINT FILE

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below
        # Store the training data
        self.X_train = X
        self.y_train = y
        # Extract unique classes
        self.classes_ = list(set(list(y)))
        return
    def _compute_distance(self, x1, x2):
            # Compute distance based on the metric
            if self.metric == "minkowski":
                return distance.minkowski(x1, x2, self.p)
            elif self.metric == "euclidean":
                return distance.euclidean(x1, x2)
            elif self.metric == "manhattan":
                return distance.cityblock(x1, x2)
            elif self.metric == "cosine":
                return 1 - distance.cosine(x1, x2)
            else:
                raise ValueError("Unsupported metric")
            
    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        predictions = []
        # For each test point, find the nearest neighbors
        for _, x_test in X.iterrows():
            # Calculate distances from the test point to all training points
            distances = [self._compute_distance(x_test, x_train) for _, x_train in self.X_train.iterrows()]
            # Get indices of the nearest neighbors
            nearest_neighbors_idx = np.argsort(distances)[:self.n_neighbors]
            # Find the labels of the nearest neighbors
            nearest_labels = [self.y_train.iloc[i] for i in nearest_neighbors_idx]
            # Get the most common label (majority voting)
            common_label = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(common_label)
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = []
        # For each test point, find the nearest neighbors and calculate probabilities
        for _, x_test in X.iterrows():
            # Calculate distances from the test point to all training points
            distances = [self._compute_distance(x_test, x_train) for _, x_train in self.X_train.iterrows()]
            # Get indices of the nearest neighbors
            nearest_neighbors_idx = np.argsort(distances)[:self.n_neighbors]
            # Find the labels of the nearest neighbors
            nearest_labels = [self.y_train.iloc[i] for i in nearest_neighbors_idx]
            # Count the occurrences of each class in the neighbors
            label_count = Counter(nearest_labels)
            # Calculate the probability for each class
            prob = {cls: label_count.get(cls, 0) / self.n_neighbors for cls in self.classes_}
            probs.append(prob)
        # Convert the list of probabilities to a DataFrame
        probs_df = pd.DataFrame(probs, columns=self.classes_)
        return probs



