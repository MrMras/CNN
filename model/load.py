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

def  load_data(path1, path2):
    # Right now instead of taking pictures from path2, takes threshold of pictures from path1
    all_items_X = os.listdir(path1)
    l = len(all_items_X)
    all_items_X = all_items_X
    all_items_Y = os.listdir(path2)

    all_items_X = group_by_n(all_items_X, config.NUM_PICS)[:-1]
    all_items_Y = group_by_n(all_items_Y, config.NUM_PICS)[:-1]
    data = list(zip(all_items_X, all_items_Y))

    np.random.shuffle(data)
    all_items_X, all_items_Y = zip(*data)
    all_items_X = np.array(all_items_X)[0:l // 2]
    all_items_Y = np.array(all_items_Y)[0:l // 2]

    # Calculate the split index based on the training ratio
    split_index = int(len(all_items_X) * config.TRAIN_RATIO)

    # Split the items into training and testing sets
    train_items_X = all_items_X[:split_index]
    test_items_X = all_items_X[split_index:]

    train_items_Y = all_items_Y[:split_index]
    test_items_Y = all_items_Y[split_index:]

    # Initialize the arrays to store training and testing data
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    # Load the data
    for array in tqdm(train_items_X, desc="Loading Training Images"):
        X_train_temp = []
        Y_train_temp = []
        for item in array:
            path = os.path.join(path1, item)
            img = plt.imread(path)
            X_train_temp.append(img)
            Y_train_temp.append((img >= 61/255).astype(np.uint8))
        X_train.append(X_train_temp)
        Y_train.append(Y_train_temp)

    for item in tqdm(test_items_X, desc="Loading Test Images"):
        X_test_temp = []
        Y_test_temp = []
        for item in array:
            path = os.path.join(path1, item)
            img = plt.imread(path)
            X_test_temp.append(img)
            Y_test_temp.append((img >= 61/255).astype(np.uint8))
        X_test.append(X_test_temp)
        Y_test.append(Y_test_temp)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)   
    
    return X_train, X_test, Y_train, Y_test
