import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # Calculate P(yj) and P(xi|yj)        
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        # write your code below
        self.class_counts_ = Counter(y)     # Count of each class in y
        self.feature_probs_ = {}            # Store feature probabilities P(xi|yj)
        self.class_probs_ = {}              # Store class probabilities P(yj)
        
        # Calculate P(yj) = N(yj) / N (prior probabilities for each class)
        total_count = len(y)
        for cls in self.classes_:
            self.class_probs_[cls] = self.class_counts_[cls] / total_count
        
        # Calculate conditional probabilities P(xi|yj) with smoothing
        for cls in self.classes_:
            subset_X = X[y == cls]  # Select all rows with class = cls
            self.feature_probs_[cls] = {}
            
            for col in X.columns:
                # Count the occurrences of each value of feature xi for class yj
                feature_counts = subset_X[col].value_counts().to_dict()
                total_feature_count = sum(feature_counts.values())
                num_categories = X[col].nunique()  # Number of unique categories for feature
                
                # Calculate P(xi|yj) with Laplace smoothing
                self.feature_probs_[cls][col] = {
                    feature_value: (feature_counts.get(feature_value, 0) + self.alpha) / 
                                   (total_feature_count + num_categories * self.alpha)
                    for feature_value in X[col].unique()
                }

        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        predictions = []
        for i in range(len(X)):
            row = X.iloc[i]
            class_scores = {}
            for cls in self.classes_:
                # Initialize with log(P(yj))
                score = np.log(self.class_probs_[cls])
                for col in X.columns:
                    # Multiply by P(xi|yj) for each feature
                    score += np.log(self.feature_probs_[cls][col].get(row[col], self.alpha / len(X)))
                class_scores[cls] = score
            # Select class with the maximum score
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)                
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # write your code below
        probs = []
        for i in range(len(X)):
            row = X.iloc[i]
            class_scores = {}
            total_score = 0
            for cls in self.classes_:
                # Initialize with log(P(yj))
                score = np.log(self.class_probs_[cls])
                for col in X.columns:
                    # Multiply by P(xi|yj) for each feature
                    score += np.log(self.feature_probs_[cls][col].get(row[col], self.alpha / len(X)))
                class_scores[cls] = np.exp(score)
                total_score += np.exp(score)
            # Normalize to get probabilities
            class_probs = {cls: class_scores[cls] / total_score for cls in class_scores}
            probs.append(class_probs)
        return pd.DataFrame(probs, columns=self.classes_)



