import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import stats

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

class KNNClassifier:
    def __init__(self, k=3, metric='euclidean', p=2):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.p = p
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(self.X_train)):
                if self.metric == 'euclidean':
                    distances.append([self.euclidean_metrics(X_test.iloc[i], self.X_train.iloc[j]), self.y_train.iloc[j]])
                elif self.metric == 'minkowski':
                    distances.append([self.minkowski_metrics(X_test.iloc[i], self.X_train.iloc[j]), self.y_train.iloc[j]])
            sorted_list = sorted(distances, key=lambda x: x[0])
            k_nearest = sorted_list[:self.k]
            mode_k_nearest = stats.mode([x[1] for x in k_nearest])
            y_pred.append(mode_k_nearest[0])

        return pd.Series(y_pred, index=X_test.index)
        
    def euclidean_metrics(self, x1, x2):
        if len(x1) != len(x2):
            return None
        else:
            sum = 0
            for i in range(len(x1)):
                sum += (x1.iloc[i] - x2.iloc[i])**2
            return sum ** 0.5

    def minkowski_metrics(self, x1, x2):
        if len(x1) != len(x2):
            return None
        else:
            sum = 0
            for i in range(len(x1)):
                sum += (abs(x1.iloc[i] - x2.iloc[i]))**self.p
            return sum ** (1/self.p)


def accuracy(y_true, y_pred):
    good_classifications = 0
    if len(y_true) != len(y_pred):
        return None
    else:
        for i in range(len(y_true)):
            if y_true.iloc[i] == y_pred[i]:
                good_classifications += 1
        return good_classifications/ len(y_true) * 100

if __name__ == "__main__":
    pass