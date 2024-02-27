#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler_instance = StandardScaler()

def scale_data(input_array):
    scaled_values = input_array.astype('float32') / 255.0
    return scaled_values

def check_scaling(input_array):
    if np.max(input_array) <= 1 and np.min(input_array) >= 0:
        return True
    else:
        return False

def check_labels(label_array):
    unique_values = np.unique(label_array)
    for val in unique_values:
        if type(val) == 'str':
            return 'String Type'
    else:
        return 'Integers'

def calculate_accuracy(confusion_matrix):
    return np.diagonal(confusion_matrix).sum() / np.sum(confusion_matrix)

def filter_90_9s(data: NDArray[np.floating], labels: NDArray[np.int32]):
    nine_indices = (labels == 9)
    data_90 = data[nine_indices, :]
    labels_90 = labels[nine_indices]

    data_90 = data_90[:int((data_90.shape[0]) * 0.1), :]
    labels_90 = labels_90[:int((labels_90.shape[0]) * 0.1)]

    non_nine_indices = (labels != 9)
    data_non_9 = data[non_nine_indices, :]
    labels_non_9 = labels[non_nine_indices]

    final_data = np.concatenate((data_non_9, data_90), axis=0)
    final_labels = np.concatenate((labels_non_9, labels_90), axis=0)

    return final_data, final_labels

def convert_7_0(data: NDArray[np.floating], labels: NDArray[np.int32]):
    id_7 = (labels == 7)
    id_0 = (labels == 0)
    labels[id_7] = 0

    return data, labels

def convert_9_1(data: NDArray[np.floating], labels: NDArray[np.int32]):
    id_9 = (labels == 9)
    id_1 = (labels == 1)
    labels[id_9] = 1

    return data, labels

