import numpy as np
import pandas as pd

def accuracy(y_true, y_pred):
    good_classifications = 0
    if len(y_true) != len(y_pred):
        return None
    else:
        for yt, yp in zip(y_true, y_pred):
            good_classifications += int(yt == yp)
        return good_classifications / len(y_true) * 100

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true))
    number_of_classes = len(classes)
    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix

def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging"""
    classes = sorted(set(y_true))
    number_of_classes = len(classes)
    conf_matrix = confusion_matrix(y_true, y_pred)
    if average == "macro":
        precision_sum = 0
        for i in range(number_of_classes):
            TP = conf_matrix[i, i]
            FP = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            if TP + FP == 0:
                precision_elem = 0
            else:
                precision_elem = TP / (TP + FP)
            precision_sum += precision_elem
        return precision_sum / number_of_classes
    elif average == "micro":
        TP_sum = 0
        for i in range(number_of_classes):
            TP = conf_matrix[i, i]
            TP_sum += TP
        FP_sum = 0
        for i in range(number_of_classes):
            FP = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            FP_sum += FP
        if TP_sum + FP_sum == 0:
            return 0
        else:
            return TP_sum / (TP_sum + FP_sum)

def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    classes = sorted(set(y_true))
    number_of_classes = len(classes)
    conf_matrix = confusion_matrix(y_true, y_pred)
    if average == "macro":
        recall_sum = 0
        for i in range(number_of_classes):
            TP = conf_matrix[i, i]
            FN = np.sum(conf_matrix[i, :]) - conf_matrix[i, i]
            if TP + FN == 0:
                recall_elem = 0
            else:
                recall_elem = TP / (TP + FN)
            recall_sum += recall_elem
        return recall_sum / number_of_classes
    elif average == "micro":
        TP_sum = 0
        for i in range(number_of_classes):
            TP = conf_matrix[i, i]
            TP_sum += TP
        FN_sum = 0
        for i in range(number_of_classes):
            FN = np.sum(conf_matrix[i, :]) - conf_matrix[i, i]
            FN_sum += FN
        if TP_sum + FN_sum == 0:
            return 0
        else:
            return TP_sum / (TP_sum + FN_sum)

def f1_score(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""
    classes = sorted(set(y_true))
    number_of_classes = len(classes)
    conf_matrix = confusion_matrix(y_true, y_pred)
    if average == "macro":
        F1_sum = 0
        for i in range(number_of_classes):
            TP = conf_matrix[i, i]
            FP = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            FN = np.sum(conf_matrix[i, :]) - conf_matrix[i, i]
            if TP + FP == 0:
                macro_precision = 0
            else:
                macro_precision = TP / (TP + FP)
            if TP + FN == 0:
                macro_recall = 0
            else:
                macro_recall = TP / (TP + FN)
            if macro_precision + macro_recall == 0:
                F1 = 0
            else:
                F1 = 2*(macro_precision*macro_recall)/(macro_precision + macro_recall)
            F1_sum += F1
        return F1_sum / number_of_classes
    elif average == "micro":
        micro_precision = precision(y_test, y_pred, 'micro')
        micro_recall = recall(y_test, y_pred, 'micro')
        return 2*(micro_precision*micro_recall)/(micro_precision+micro_recall)  
