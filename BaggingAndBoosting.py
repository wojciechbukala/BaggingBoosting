import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# self implemented models
from kNNClassifier import KNNClassifier
from OVR import OVRClassifier

def data_analize(X, y):
    pd.set_option('display.max_columns', None)
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Liczba klas: {len(set(y))}")
    print(f"Liczność klas \n{y.value_counts()}")

    print('--- Row data ---')
    print(X.describe().round(2))
    print('----------------')

if __name__ == "__main__":
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    data_analize(X, y)
