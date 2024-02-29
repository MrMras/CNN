import config
import torch
import numpy as np
import os

from copy import deepcopy
from tqdm import tqdm

def train(model, criterion, optimizer, train_loader, device, epochs, weights):
    loss_curve = []
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
            # loss = (loss * (weights[0] + labels * (weights[1] - weights[0]))).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Print the average loss for this epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}\n")
        loss_curve.append(running_loss / len(train_loader))
    # Get a random id from 10^6 to 10^7 - 1
    model_id = np.random.randint(1000000, 10000000 - 1)
    # Saving the model's state dictionary
    if not os.path.exists("../saved_models"):
        os.makedirs("../saved_models")
    torch.save(deepcopy(model).cpu().state_dict(), f'../saved_models/model_for_vasc_3d{model_id}.pth')
    print(f"Model saved as model_for_vasc_3d{model_id}.pth")
    # Set the model to evaluation mode
    return model, loss_curve
