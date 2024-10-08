import pandas as pd
import numpy as np
from collections import Counter
#DID NOT USE HINT FILE
class my_DT:

    def __init__(self, criterion="gini", max_depth=8, min_impurity_decrease=0, min_samples_split=2):
        # criterion = {"gini", "entropy"},
        # Stop training if depth = max_depth. Depth of a binary tree: the max number of edges from the root node to a leaf node
        # Only split node if impurity decrease >= min_impurity_decrease after the split
        #   Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        # Only split node with >= min_samples_split samples
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = int(min_samples_split)
        
        #HELPER METHODS
    def gini(self, y):
            # Calculate Gini impurity
        counts = Counter(y)
        impurity = 1
        total = len(y)
        for label in counts:
            prob_of_label = counts[label] / total
            impurity -= prob_of_label ** 2
        return impurity

    def entropy(self, y):
        # Calculate Entropy
        counts = Counter(y)
        total = len(y)
        entropy = 0
        for label in counts:
            prob_of_label = counts[label] / total
            entropy -= prob_of_label * np.log2(prob_of_label)
        return entropy

    def best_split(self, X, y):
        # Finding the best feature and threshold to split the data
        best_gain = 0
        best_split = None
        current_impurity = self.gini(y) if self.criterion == "gini" else self.entropy(y)
        
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            thresholds = X.iloc[:, feature_idx].unique()
            for threshold in thresholds:
                left_mask = X.iloc[:, feature_idx] < threshold
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                left_impurity = self.gini(left_y) if self.criterion == "gini" else self.entropy(left_y)
                right_impurity = self.gini(right_y) if self.criterion == "gini" else self.entropy(right_y)
                
                N = len(y)
                N_left = len(left_y)
                N_right = len(right_y)
                
                weighted_impurity = (N_left / N) * left_impurity + (N_right / N) * right_impurity
                impurity_decrease = current_impurity - weighted_impurity
                
                if impurity_decrease >= self.min_impurity_decrease and impurity_decrease > best_gain:
                    best_gain = impurity_decrease
                    best_split = {
                        "feature_idx": feature_idx,
                        "threshold": threshold,
                        "left_indices": left_mask,
                        "right_indices": right_mask
                    }
        return best_split

    def build_tree(self, X, y, depth):
        # Recursively building the decision tree
        if len(y) < self.min_samples_split or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]  # Return majority class at leaf node
        
        split = self.best_split(X, y)
        if split is None:
            return Counter(y).most_common(1)[0][0]  # Return majority class if no split
        
        left_tree = self.build_tree(X[split["left_indices"]], y[split["left_indices"]], depth + 1)
        right_tree = self.build_tree(X[split["right_indices"]], y[split["right_indices"]], depth + 1)
        
        return {
            "feature_idx": split["feature_idx"],
            "threshold": split["threshold"],
            "left_tree": left_tree,
            "right_tree": right_tree
        }
    #END OF HELPER METHODS
    
    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below
        self.tree = self.build_tree(X, y, 0)
        return
    
    def classify(self, node, row):
        # Traverse the tree and classify based on features
        if isinstance(node, dict):
            if row.iloc[node["feature_idx"]] < node["threshold"]:
                return self.classify(node["left_tree"], row)
            else:
                return self.classify(node["right_tree"], row)
        else:
            return node


    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        # Predict class labels for each data point
        predictions = [self.classify(self.tree, row) for _, row in X.iterrows()]
        return predictions
    
    def classify_proba(self, node, row):
        # Traverse the tree and return the class probabilities at the leaf
        if isinstance(node, dict):
            if row.iloc[node["feature_idx"]] < node["threshold"]:
                return self.classify_proba(node["left_tree"], row)
            else:
                return self.classify_proba(node["right_tree"], row)
        else:
            return {class_: 1 if class_ == node else 0 for class_ in self.classes_}


    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # Eample:
        # self.classes_ = {"2", "1"}
        # the reached node for the test data point has {"1":2, "2":1}
        # then the prob for that data point is {"2": 1/3, "1": 2/3}
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = []
        for _, row in X.iterrows():
            class_probs = self.classify_proba(self.tree, row)
            total = sum(class_probs.values())
            prob = {class_: count / total for class_, count in class_probs.items()}
            probs.append(prob)
        ##################
        return pd.DataFrame(probs, columns=self.classes_)



