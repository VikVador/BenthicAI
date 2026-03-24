import matplotlib.pyplot as plt
import numpy as np



def bias(y_true, y_pred):
    return y_pred - y_true


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_true_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum((y_true - y_true_mean) ** 2, axis=0)
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return np.sqrt((y_true - y_pred)**2)



