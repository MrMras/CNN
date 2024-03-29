#Import the libraries

# import cv2
import sys
sys.path.append("../")
import config

from load import load_data
from train import train
from tverskyLoss import TverskyLoss
from tverskyLoss import FocalTverskyLoss

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim


from copy import deepcopy
# from cv2 import imread, imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE
from get_stats import get_stats
from model_structures.model_structure_3d_1l import UNet3D
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Initalize the seed for the possibility to repeat the result
seed = 1
np.random.seed = seed

# Images dimensions
IMG_WIDTH = config.WIDTH
IMG_HEIGHT = config.HEIGHT
IMG_CHANNELS = 1

# Define the path to data folder
IN_DATA_NPY = "../data/micro_ct/volume_input.npy"
OUT_DATA_NPY = "../data/micro_ct/volume_ground_truth.npy"

X_train, X_test, Y_train, Y_test = load_data(IN_DATA_NPY, OUT_DATA_NPY)

# b = random.randint(0, len(X_train))
# image_index = b
# imshow(X_train[image_index])
# plt.show()
# imshow(np.squeeze(Y_train[image_index]))
# plt.show()

# Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}.")

# Create the model
model = UNet3D().to(device)
print("Model created.")

ratio = (np.sum(Y_train==0)+1) / (np.sum(Y_train==1) + 1)
weights = [1 / (1 + ratio), ratio / (1 + ratio)]
weights = [0.5, 0.5]
print(f"Ratio {weights[0]} to {weights[1]}")

# Define loss function and optimizer
criterion = FocalTverskyLoss(weights[0], weights[1])  # Binary Cross-Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Criterion and optimizer created.")

# Add a channel dimension (1 channel)
X_train_tensor = torch.Tensor(X_train).unsqueeze(1).to(device)  
Y_train_tensor = torch.Tensor(Y_train).unsqueeze(1).to(device)
print("Unsqueezed.")

# Create a DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print("Loaders created.")

# Train the modeil
tag = 'a'
if len(sys.argv) == 1:
    epochs = 20
elif len(sys.argv) == 2:
    epochs = int(sys.argv[1])
else:
    epochs = int(sys.argv[1])
    tag = 'w'

dataset_name = IN_DATA_NPY.split("/")[-2]
model, positive_accuracy, negative_accuracy = train(model, criterion, optimizer, train_loader, device, epochs, dataset_name)

# Specify the file name
file_name1 = 'positive_accuracy.txt'
file_name2 = 'negative_accuracy.txt'

# Save the loss_curve to the file
with open(file_name1, tag) as file:
    np.savetxt(file, [positive_accuracy], newline='\n')
    
with open(file_name2, tag) as file:
    np.savetxt(file, [negative_accuracy], newline='\n')

get_stats(model, X_test, Y_test, device)
