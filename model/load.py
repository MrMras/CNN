import config
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from tqdm import tqdm

def group_by_n(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def load_data(in_path, out_path):
    # Load the input and output data
    all_items_X = np.load(in_path)
    all_items_Y = np.load(out_path)
    data = list(zip(all_items_X, all_items_Y))

    # Shuffle the data
    np.random.shuffle(data)
    all_items_X, all_items_Y = zip(*data)

    # Normalize the data
    x_min, x_max, y_min, y_max = np.min(all_items_X), np.max(all_items_X), np.min(all_items_Y), np.max(all_items_Y)
    all_items_X = (np.array(all_items_X) - x_min) / (x_max - x_min)
    all_items_Y = (np.array(all_items_Y) - y_min) / (y_max - y_min)

    # Calculate the split index based on the training ratio
    split_index = int(len(all_items_X) * config.TRAIN_RATIO)

    # Split the items into training and testing sets
    train_items_X = all_items_X[:split_index]
    test_items_X = all_items_X[split_index:]

    train_items_Y = all_items_Y[:split_index]
    test_items_Y = all_items_Y[split_index:]

    return train_items_X, test_items_X, train_items_Y, test_items_Y

