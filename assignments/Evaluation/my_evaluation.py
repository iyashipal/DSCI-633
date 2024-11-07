import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score

#DID NOT USE HINT FILE

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        # Initialize confusion matrix for each class
        self.confusion_matrix = {}
        for cls in self.classes_:
            TP = sum((self.predictions == cls) & (self.actuals == cls))
            FP = sum((self.predictions == cls) & (self.actuals != cls))
            FN = sum((self.predictions != cls) & (self.actuals == cls))
            TN = sum((self.predictions != cls) & (self.actuals != cls))
            self.confusion_matrix[cls] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}



    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below

        if self.confusion_matrix==None:
            self.confusion()

        if target:
            # Recall for a specific class
            TP = self.confusion_matrix[target]["TP"]
            FP = self.confusion_matrix[target]["FP"]
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            return prec
        else:
            # Average precision
            precisions = []
            for cls in self.classes_:
                precisions.append(self.precision(target=cls))
            if average == "macro":
                return np.mean(precisions)
            elif average == "weighted":
                weights = [sum(self.actuals == cls) for cls in self.classes_]
                return np.average(precisions, weights=weights)
            else:  # micro
                total_TP = sum([self.confusion_matrix[cls]["TP"] for cls in self.classes_])
                total_FP = sum([self.confusion_matrix[cls]["FP"] for cls in self.classes_])
                return total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0.0



    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        if target:
            # Recall for a specific class
            TP = self.confusion_matrix[target]["TP"]
            FN = self.confusion_matrix[target]["FN"]
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            return rec
        else:
            # Average recall
            recalls = []
            for cls in self.classes_:
                recalls.append(self.recall(target=cls))
            if average == "macro":
                return np.mean(recalls)
            elif average == "weighted":
                weights = [sum(self.actuals == cls) for cls in self.classes_]
                return np.average(recalls, weights=weights)
            else:  # micro
                total_TP = sum([self.confusion_matrix[cls]["TP"] for cls in self.classes_])
                total_FN = sum([self.confusion_matrix[cls]["FN"] for cls in self.classes_])
                return total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0.0



    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        # F1 score is the harmonic mean of precision and recall
        if target:
            prec = self.precision(target=target)
            rec = self.recall(target=target)
            return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        else:
            # Average F1
            f1_scores = []
            for cls in self.classes_:
                f1_scores.append(self.f1(target=cls))
            if average == "macro":
                return np.mean(f1_scores)
            elif average == "weighted":
                weights = [sum(self.actuals == cls) for cls in self.classes_]
                return np.average(f1_scores, weights=weights)
            else:  # micro
                prec_micro = self.precision(average="micro")
                rec_micro = self.recall(average="micro")
                return (2 * prec_micro * rec_micro) / (prec_micro + rec_micro)        


    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba) == type(None):
            return None
        else:
            # write your own code below
            if target in self.pred_proba.columns:
                actuals_binary = (self.actuals == target).astype(int)
                return roc_auc_score(actuals_binary, self.pred_proba[target])
            else:
                return None
            


