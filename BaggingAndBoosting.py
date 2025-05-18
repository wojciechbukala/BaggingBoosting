import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import time

# sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# self implemented models
from kNNClassifier import KNNClassifier
from OVR import OVRClassifier

from metrics import confusion_matrix, accuracy, precision, recall ,f1_score

def data_analize(X, y):
    pd.set_option('display.max_columns', None)
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Liczba klas: {len(set(y))}")
    print(f"Liczność klas \n{y.value_counts()}")

    print('--- Row data ---')
    print(X.describe().round(2))
    print('----------------')

def data_standardize(X):
    return (X - X.mean()) / X.std()

def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def run_knn(X_train, X_test, y_train):
    KNN = KNNClassifier()
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    return y_pred

def run_ovr(X_train, X_test, y_train):
    OVR = OVRClassifier()
    OVR.fit(X_train, y_train)
    y_pred = OVR.predict(X_test)
    return y_pred

def run_random_forest(X_train, X_test, y_train):
    RANDFOREST = RandomForestClassifier()
    RANDFOREST.fit(X_train, y_train)
    y_pred = RANDFOREST.predict(X_test)
    return pd.Series(y_pred, index=X_test.index)

def run_bagging_classifier(X_train, X_test, y_train):
    BAGGINGCLASSIFIER = BaggingClassifier()
    BAGGINGCLASSIFIER.fit(X_train, y_train)
    y_pred = BAGGINGCLASSIFIER.predict(X_test)
    return pd.Series(y_pred, index=X_test.index)

def run_ada_boost(X_train, X_test, y_train):
    ADABOOST = AdaBoostClassifier()
    ADABOOST.fit(X_train, y_train)
    y_pred = ADABOOST.predict(X_test)
    return pd.Series(y_pred, index=X_test.index)

if __name__ == "__main__":
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    #data_analize(X, y)

    X = data_standardize(X)

    #data_analize(X, y)

    X_train, X_test, y_train, y_test = data_split(X, y)

    # --- kNN Classifier ---
    print("--- kNN Classifier ---")
    start = time.perf_counter()
    y_pred = run_knn(X_train, X_test, y_train)
    end = time.perf_counter()
    print(confusion_matrix(y_test, y_pred)) # confusion matrix
    print(f'Accuracy: {accuracy(y_test, y_pred)}')
    print(f'Precision (macro): {precision(y_test, y_pred)}')
    print(f'Precision (micro): {precision(y_test, y_pred, "micro")}')
    print(f'Recall (macro): {recall(y_test, y_pred)}')
    print(f'Recall (micro): {recall(y_test, y_pred, "micro")}')
    print(f'F1 score (macro): {f1_score(y_test, y_pred)}')
    print(f'F1 score (micro): {f1_score(y_test, y_pred, "micro")}')
    print(f'Processing time: {end - start:.4f}')
    print()

    # --- OVR Perceptron ---
    print("--- OVR Perceptron ---")
    start = time.perf_counter()
    y_pred = run_ovr(X_train, X_test, y_train)
    end = time.perf_counter()
    print(confusion_matrix(y_test, y_pred)) # confusion matrix
    print(f'Accuracy: {accuracy(y_test, y_pred)}')
    print(f'Precision (macro): {precision(y_test, y_pred)}')
    print(f'Precision (micro): {precision(y_test, y_pred, "micro")}')
    print(f'Recall (macro): {recall(y_test, y_pred)}')
    print(f'Recall (micro): {recall(y_test, y_pred, "micro")}')
    print(f'F1 score (macro): {f1_score(y_test, y_pred)}')
    print(f'F1 score (micro): {f1_score(y_test, y_pred, "micro")}')
    print(f'Processing time: {end - start:.4f}')
    print()

    # --- Random forest ---
    print("--- Random forest classifier ---")
    start = time.perf_counter()
    y_pred = run_random_forest(X_train, X_test, y_train)
    end = time.perf_counter()
    print(confusion_matrix(y_test, y_pred)) # confusion matrix
    print(f'Accuracy: {accuracy(y_test, y_pred)}')
    print(f'Precision (macro): {precision(y_test, y_pred)}')
    print(f'Precision (micro): {precision(y_test, y_pred, "micro")}')
    print(f'Recall (macro): {recall(y_test, y_pred)}')
    print(f'Recall (micro): {recall(y_test, y_pred, "micro")}')
    print(f'F1 score (macro): {f1_score(y_test, y_pred)}')
    print(f'F1 score (micro): {f1_score(y_test, y_pred, "micro")}')
    print(f'Processing time: {end - start:.4f}')
    print()

    # --- Bagging Classifier ---
    print("--- Bagging Classifier ---")
    start = time.perf_counter()
    y_pred = run_bagging_classifier(X_train, X_test, y_train)
    end = time.perf_counter()
    print(confusion_matrix(y_test, y_pred)) # confusion matrix
    print(f'Accuracy: {accuracy(y_test, y_pred)}')
    print(f'Precision (macro): {precision(y_test, y_pred)}')
    print(f'Precision (micro): {precision(y_test, y_pred, "micro")}')
    print(f'Recall (macro): {recall(y_test, y_pred)}')
    print(f'Recall (micro): {recall(y_test, y_pred, "micro")}')
    print(f'F1 score (macro): {f1_score(y_test, y_pred)}')
    print(f'F1 score (micro): {f1_score(y_test, y_pred, "micro")}')
    print(f'Processing time: {end - start:.4f}')
    print()

    # --- AdaBoost Classifier ---
    print("--- AdaBoost Classifier ---")
    start = time.perf_counter()
    y_pred = run_ada_boost(X_train, X_test, y_train)
    end = time.perf_counter()
    print(confusion_matrix(y_test, y_pred)) # confusion matrix
    print(f'Accuracy: {accuracy(y_test, y_pred)}')
    print(f'Precision (macro): {precision(y_test, y_pred)}')
    print(f'Precision (micro): {precision(y_test, y_pred, "micro")}')
    print(f'Recall (macro): {recall(y_test, y_pred)}')
    print(f'Recall (micro): {recall(y_test, y_pred, "micro")}')
    print(f'F1 score (macro): {f1_score(y_test, y_pred)}')
    print(f'F1 score (micro): {f1_score(y_test, y_pred, "micro")}')
    print(f'Processing time: {end - start:.4f}')
    print()
    


