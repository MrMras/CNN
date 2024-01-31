#Import the libraries
# import cv2
import config
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
# from cv2 import imread, imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE
from model_structure_3d import UNet3D as UNet
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
IN_DATA_PATH = "./dataset/indata/"
OUT_DATA_PATH = "./dataset/outdata/"

# Define the ratio of data to be used for training (e.g., 80%)
TRAIN_RATIO = 0.75

# List all items (files) in the data folder
all_items_X = os.listdir(IN_DATA_PATH)
l = len(all_items_X)
all_items_X = all_items_X
all_items_Y = os.listdir(OUT_DATA_PATH)

# Shuffle the list of items randomly, without losing the connection between 
# Function for grouping the elements
def group_by_n(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]
all_items_X = group_by_n(all_items_X, config.NUM_PICS)[:-1]
all_items_Y = group_by_n(all_items_Y, config.NUM_PICS)[:-1]
data = list(zip(all_items_X, all_items_Y))

np.random.shuffle(data)
all_items_X, all_items_Y = zip(*data)
all_items_X = np.array(all_items_X)[0:l // 2]
all_items_Y = np.array(all_items_Y)[0:l // 2]

# Calculate the split index based on the training ratio
split_index = int(len(all_items_X) * TRAIN_RATIO)

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
        path = os.path.join(IN_DATA_PATH, item)
        img = plt.imread(path)
        X_train_temp.append(img / 255)
        Y_train_temp.append((img >= 61).astype(np.uint8))
    X_train.append(X_train_temp)
    Y_train.append(Y_train_temp)

for item in tqdm(test_items_X, desc="Loading Test Images"):
    X_test_temp = []
    Y_test_temp = []
    for item in array:
        path = os.path.join(IN_DATA_PATH, item)
        img = plt.imread(path)
        X_test_temp.append(img / 255)
        Y_test_temp.append((img >= 61).astype(np.uint8))
    X_test.append(X_test_temp)
    Y_test.append(Y_test_temp)

# Convert lists to NumPy arrays
X_train = np.array(X_train)
print("X_train shape:", X_train.shape)

X_test = np.array(X_test)
print("X_test shape:", X_test.shape)

Y_train = np.array(Y_train)
print("Y_train shape:", Y_train.shape)

Y_test = np.array(Y_test)
print("Y_test shape:", Y_test.shape)

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
model = UNet().to(device)
print("Model created.")

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary segmentation
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

weights = [1, 11]
# Training loop
epochs = int(input("Enter preffered epoch count: "))
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    i = 0
    
    # Wrap train_loader with tqdm for a progress bar
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = (loss * (weights[0] + labels * (weights[1] - weights[0]))).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Print the average loss for this epoch
    print(f"\nEpoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
# Saving the model's state dictionary
torch.save(deepcopy(model).cpu().state_dict(), 'model_for_vasc_3d.pth')

# Set the model to evaluation mode
model.eval()

# Initialize an empty list to store the predictions
preds_test = []

# Convert X_test to a tensor and move it to the appropriate device
X_test_tensor = torch.Tensor(X_test).unsqueeze(1).to(device)
print("X_test_tensor shape:", X_test_tensor.shape)

# Perform forward pass without gradient computation
with torch.no_grad():
    outputs = model(X_test_tensor)
    preds_test.append(outputs.cpu().numpy())

# Convert predictions to binary masks
preds_test = (np.concatenate(preds_test) > 0.5).astype(np.uint8)

# Convert predictions to binary masks
preds_test_binary = (preds_test > 0.5).astype(np.uint8)

# Flatten the ground truth masks (Y_test) and predicted masks (preds_test_binary)
y_true = Y_test.flatten()
y_pred = preds_test_binary.flatten()

# Calculate TP, FP, TN, FN
TP = np.sum((y_true == 1) & (y_pred == 1))
FP = np.sum((y_true == 0) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FN = np.sum((y_true == 1) & (y_pred == 0))

# Display TP, FP, TN, FN
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)
print("False Negatives (FN):", FN)

# Calculate other parameters
print("Positive's accuracy:", TP / (TP + FN))
print("Negative's accuracy:", TN / (TN + FP))

print("Positive's recall:", TP / (TP + FP))
print("Negative's recaLL:", TN / (TN + FN))

print("Total accuracy:", (TP + TN) / (TP + TN + FP + FN))
print("Weighted accuracy (multiplier = 4):", (4 * TP + TN) / (4 * TP + TN + FP + 4 * FN))
