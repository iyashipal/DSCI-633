from random import sample
import numpy as np
from scipy.linalg import svd
from copy import deepcopy

# DID NOT USE my_preprocess_hint.py

class my_normalizer:
    def __init__(self, norm="Min-Max", axis = 1):
        #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
        #     axis = 0: normalize rows
        #     axis = 1: normalize columns
        self.norm = norm
        self.axis = axis

    def fit(self, X):
        #     X: input matrix
        #     Calculate offsets and scalers which are used in transform()
        X_array  = np.asarray(X)
        # Write your own code below
        if self.norm == "Min-Max":
            # Calculate min and max for each row or column
            self.min_ = X_array.min(axis=self.axis, keepdims=True)
            self.max_ = X_array.max(axis=self.axis, keepdims=True)

        elif self.norm == "Standard_Score":
            # Calculate mean and std for each row or column
            self.mean_ = X_array.mean(axis=self.axis, keepdims=True)
            self.std_ = X_array.std(axis=self.axis, keepdims=True)


    def transform(self, X):
        # Transform X into X_norm
        X_norm = deepcopy(np.asarray(X))
        # Write your own code below
        if self.norm == "Min-Max":
            X_norm = (X_norm - self.min_) / (self.max_ - self.min_)
        elif self.norm == "Standard_Score":
            X_norm = (X_norm - self.mean_) / self.std_
        elif self.norm == "L1":
            # L1 normalization: Sum of absolute values of rows or columns should be 1
            norm = np.sum(np.abs(X_norm), axis=self.axis, keepdims=True)
            X_norm = X_norm / norm
        elif self.norm == "L2":
            # L2 normalization: Sum of squares of rows or columns should be 1
            norm = np.sqrt(np.sum(np.square(X_norm), axis=self.axis, keepdims=True))
            X_norm = X_norm / norm

        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class my_pca:
    def __init__(self, n_components = 5):
        #     n_components: number of principal components to keep
        self.n_components = n_components

    def fit(self, X):
        #  Use svd to perform PCA on X
        #  Inputs:
        #     X: input matrix
        #  Calculates:
        #     self.principal_components: the top n_components principal_components
        # Vh = the transpose of V
        U, s, Vh = svd(X)
        # Write your own code below
         # Keep the first n_components from Vh (transpose of V)
        self.principal_components = Vh[:self.n_components].T

    def transform(self, X):
        #     X_pca = X.dot(self.principal_components)
        X_array = np.asarray(X)
        # Write your own code below
        # Project X onto the principal components
        X_pca = X_array.dot(self.principal_components)
        return X_pca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: len(sample) = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    unique_classes, class_counts = np.unique(y_array, return_counts=True)
    sample = []
    y_array = np.asarray(y)
    # Write your own code below
     # Perform stratified sampling for each class
    for cls, count in zip(unique_classes, class_counts):
        n_samples = int(np.ceil(ratio * count))
        class_indices = np.where(y_array == cls)[0]
        class_sample = np.random.choice(class_indices, size=n_samples, replace=replace)
        sample.extend(class_sample)


    return np.array(sample).astype(int)
