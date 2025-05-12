import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X_train, y_train):
        self.bias = 0
        self.weights = np.zeros(len(X_train.columns))
        for i in range(self.epochs):
            for index, row in X_train.iterrows():
                while True: 
                    scalar = np.dot(row, self.weights) + self.bias
                    real_class = y_train.loc[index]
                    pred_class = int(scalar >= 0)
    
                    if pred_class != real_class:
                        self.weights += (self.learning_rate * (real_class - pred_class)) * row
                        self.bias += (self.learning_rate * (real_class - pred_class))  
                    else:
                        break

    def predict_proba(self, X_test):
        y_pred_scalars = pd.Series()
        for index, row in X_test.iterrows():
            scalar = np.dot(row, self.weights) + self.bias
            y_pred_scalars.loc[index] = scalar
        return y_pred_scalars

    def predict(self, X_test):
        y_pred = pd.Series()
        for index, row in X_test.iterrows():
            scalar = np.dot(row, self.weights) + self.bias
            pred_class = int(scalar >= 0)
            y_pred.loc[index] = pred_class
        return y_pred