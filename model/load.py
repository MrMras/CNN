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

    np.random.shuffle(data)
    all_items_X, all_items_Y = zip(*data)
    all_items_X = np.array(all_items_X) / 255
    all_items_Y = np.array(all_items_Y) // 255
    print(np.max(all_items_X))

    # Calculate the split index based on the training ratio
    split_index = int(len(all_items_X) * config.TRAIN_RATIO)

    # Split the items into training and testing sets
    train_items_X = all_items_X[:split_index]
    test_items_X = all_items_X[split_index:]

    train_items_Y = all_items_Y[:split_index]
    test_items_Y = all_items_Y[split_index:]

    return train_items_X, test_items_X, train_items_Y, test_items_Y

    # # Initialize the arrays to store training and testing data
    # X_train = []
    # X_test = []
    # Y_train = []
    # Y_test = []

    # # Load the data
    # for array in tqdm(train_items_X, desc="Loading Training Images"):
    #     X_train_temp = []
    #     Y_train_temp = []
    #     for item in array:
    #         path_in = os.path.join(path1, item)
    #         path_out = os.path.join(path2, item)
    #         img_in = plt.imread(path_in)
    #         img_out = plt.imread(path_out)
    #         X_train_temp.append(img_in)
    #         Y_train_temp.append((img_in > 61 / 255).astype(np.uint8))
    #     X_train.append(X_train_temp)
    #     Y_train.append(Y_train_temp)

    # for item in tqdm(test_items_X, desc="Loading Test Images"):
    #     X_test_temp = []
    #     Y_test_temp = []
    #     for item in array:
    #         path_in = os.path.join(path1, item)
    #         path_out = os.path.join(path2, item)
    #         img_in = plt.imread(path_in)
    #         img_out = plt.imread(path_out)
    #         X_test_temp.append(img_in)
    #         Y_test_temp.append((img_in > 61 / 255).astype(np.uint8))
    #     X_test.append(X_test_temp)
    #     Y_test.append(Y_test_temp)
    
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)

    # Y_train = np.array(Y_train)
    # Y_test = np.array(Y_test)   
    
    # return X_train, X_test, Y_train, Y_test
