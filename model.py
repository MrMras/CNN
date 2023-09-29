#Import the libraries
import os
import random
import numpy as np
import config
from cv2 import imread, imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import threading


# Initalize the seed for the possibility to repeat the result
seed = 42
np.random.seed = seed

# Images dimensions
IMG_WIDTH = config.WIDTH
IMG_HEIGHT = config.HEIGHT
IMG_CHANNELS = 1

# Define the path to data folder
IN_DATA_PATH = "./dataset/indata/"
OUT_DATA_PATH = "./dataset/outdata/"

# Define the ratio of data to be used for training (e.g., 80%)
TRAIN_RATIO = 0.8

# List all items (files) in the data folder
all_items_X = os.listdir(IN_DATA_PATH)
all_items_Y = os.listdir(OUT_DATA_PATH)

# Shuffle the list of items randomly, without losing the connection between 
data = list(zip(all_items_X, all_items_Y))
np.random.shuffle(data)
all_items_X, all_items_Y = zip(*data)
all_items_X = np.array(all_items_X)
all_items_Y = np.array(all_items_Y)

# Calculate the split index based on the training ratio
split_index = int(len(all_items_X) * TRAIN_RATIO)

# Split the items into training and testing sets
train_items_X = all_items_X[:split_index]
test_items_X = all_items_X[split_index:]

train_items_Y = all_items_Y[:split_index]
test_items_Y = all_items_Y[split_index:]

# Initialize arrays to store training and testing data
X_train = []
X_test = []
Y_train = []
Y_test = []

# Load the data
print('Loading training images...')
for item in train_items_X:
    path = os.path.join(IN_DATA_PATH, item)
    img = imread(path, IMREAD_GRAYSCALE)
    X_train.append(img / 255)

print('Loading test images...')
for item in test_items_X:
    path = os.path.join(IN_DATA_PATH, item)
    img = imread(path, IMREAD_GRAYSCALE)
    X_test.append(img / 255)

print('Loading training masks...')
for item in train_items_Y:
    path = os.path.join(OUT_DATA_PATH, item)
    img = imread(path, IMREAD_GRAYSCALE)
    Y_train.append(img / 255)

print('Loading test masks...')
for item in test_items_Y:
    path = os.path.join(OUT_DATA_PATH, item)
    img = imread(path, IMREAD_GRAYSCALE)
    Y_test.append(img / 255)
print('Done!')

# Convert lists to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


# b = random.randint(0, len(X_train))
# image_index = b
# imshow(X_train[image_index])
# plt.show()
# imshow(np.squeeze(Y_train[image_index]))
# plt.show()


# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contraction Path (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        # Expansion Path (Decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )

        # Final output layer
        self.out_conv = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc_features = self.encoder(x)
        
        # Decoder
        dec_features = self.decoder(enc_features)
        
        # Final output
        out = self.out_conv(dec_features)
        out = self.sigmoid(out)
        
        return out


# Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
model = UNet().to(device)
print("Model created.")

# Print the model summary
print(model)

# Create the model
model = UNet()
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


# Training loop
epochs = 5
print("." * 100)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    i = 0
    threads = []

    for inputs, labels in train_loader:
        if i % 32 == 0:
            print(".", end="")
        i += 1

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Print the average loss for this epoch
    print(f"\nEpoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Saving the model's state dictionary
torch.save(model.state_dict(), 'model_for_vasc.pth')

# Validation and prediction
model.eval()  # Set the model to evaluation mode
preds_train = []
preds_test = []

with torch.no_grad():
    for inputs, _ in train_loader:
        outputs = model(inputs)
        preds_train.append(outputs.numpy())

X_test_tensor = torch.Tensor(X_test).unsqueeze(1)

with torch.no_grad():
    outputs = model(X_test_tensor)
    preds_test.append(outputs.numpy())

# Convert predictions to binary masks
preds_train = (np.concatenate(preds_train) > 0.5).astype(np.uint8)
preds_test = (np.concatenate(preds_test) > 0.5).astype(np.uint8)

# Perform a sanity check on random training samples
ix = random.randint(0, len(preds_train))
plt.imshow(X_train[ix].squeeze(), cmap='gray')
plt.show()
plt.imshow(Y_train[ix].squeeze(), cmap='gray')
plt.show()
plt.imshow(preds_train[ix].squeeze(), cmap='gray')
plt.show()
