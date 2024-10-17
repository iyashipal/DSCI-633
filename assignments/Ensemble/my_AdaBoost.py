import pandas as pd
import numpy as np
from copy import deepcopy
#DID NOT USE HINT FILE
class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):
        # Multi-class Adaboost algorithm (SAMME)
        # alpha = ln((1-error)/error)+ln(K-1), K is the number of classes.
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]
        self.alphas = []  # Store alpha values (weights of each weak learner)

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        # write your code below
        # Ensure X and y are compatible
        self.classes_ = list(set(y))  # List of unique classes
        k = len(self.classes_)  # Number of unique classes
        n_samples = X.shape[0]  # Number of samples

        # Initialize sample weights evenly
        w = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            # Train the base estimator with feature names preserved
            self.estimators[i].fit(X, y, sample_weight=w)
            predictions = self.estimators[i].predict(X)

        # Calculate the weighted error rate
        incorrect = (predictions != y).astype(int)
        error = np.dot(w, incorrect) / np.sum(w)

        # Compute alpha (model weight)
        alpha = np.log((1 - error) / (error + 1e-10)) + np.log(k - 1)
        self.alphas.append(alpha)

        # Update sample weights
        w *= np.exp(alpha * incorrect)
        w /= np.sum(w)  # Normalize the weights

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        # Initialize an array to store class scores
        class_scores = np.zeros((X.shape[0], len(self.classes_)))

        # Sum the weighted predictions from each weak learner
        for alpha, estimator in zip(self.alphas, self.estimators):
            predictions = estimator.predict(X)  # Use DataFrame directly
            for i, pred in enumerate(predictions):
                class_idx = self.classes_.index(pred)
                class_scores[i, class_idx] += alpha

        # Choose the class with the highest score for each sample
        predictions = [self.classes_[np.argmax(score)] for score in class_scores]
        return predictions


    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        # Compute probabilities
        probs = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, estimator in zip(self.alphas, self.estimators):
            predictions = estimator.predict(X)  # Use DataFrame directly
            for i, pred in enumerate(predictions):
                class_idx = self.classes_.index(pred)
                probs[i, class_idx] += alpha

        # Normalize probabilities
        probs /= np.sum(probs, axis=1, keepdims=True)

        # Return as DataFrame with class names as columns
        return pd.DataFrame(probs, columns=self.classes_)





