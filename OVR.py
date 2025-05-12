import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy import stats

from perceptron import Perceptron

from sklearn.svm import SVC


class OVRClassifier():
    def __init__(self, model='perceptron', learning_rate=0.1, epochs=10):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.models = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        classes = sorted(y_train.unique())
        self.models = []

        for y_class in classes:
            y_binary = (y_train == y_class).astype(int)

            if self.model == 'perceptron':
                model_instance = Perceptron(self.learning_rate, self.epochs)
            elif self.model == 'svc':
                model_instance = SVC(kernel="linear", probability=True)
            else:
                raise ValueError(f"Unknown model: {self.model}")

            model_instance.fit(X_train, y_binary)
            self.models.append(model_instance)

    def predict(self, X_test: pd.DataFrame):
        if self.model == 'perceptron':
            predictions_array = np.array([model.predict_proba(X_test) for model in self.models])
        elif self.model == 'svc':
            predictions_array = np.array([model.decision_function(X_test) for model in self.models])
        else:
            raise ValueError("Unknown model")

        y_pred_list = [np.argmax(predictions_array[:, i]) for i in range(predictions_array.shape[1])]
        return pd.Series(y_pred_list, index=X_test.index)