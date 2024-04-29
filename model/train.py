import config
import torch
import numpy as np
import os

from copy import deepcopy
from tqdm import tqdm

def train(model, criterion, optimizer, train_loader, device, epochs, dataset_name):
    positive_accuracy = []
    negative_accuracy = []
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

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Calculate accuracy after each epoch
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Assuming binary classification (positive and negative classes)
            predictions = (outputs > 0.5).float()  # Assuming a threshold of 0.5 for binary classification
            correct_predictions = predictions.eq(labels.view_as(predictions)).sum().item()
            total_samples = labels.size(0)

            positive_mask = labels == 1 
            negative_mask = labels == 0 

            positive_correct = (predictions[positive_mask] == 1).sum().item()
            negative_correct = (predictions[negative_mask] == 0).sum().item()

            positive_accuracy_epoch = positive_correct / max(positive_mask.sum().item(), 1)  # Avoid division by zero
            negative_accuracy_epoch = negative_correct / max(negative_mask.sum().item(), 1)  # Avoid division by zero

            positive_accuracy.append(positive_accuracy_epoch)
            negative_accuracy.append(negative_accuracy_epoch)

        model.train()  # Set the model back to training mode

        # Print the average loss for this epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}\n")

    # Get a random id from 10^6 to 10^7 - 1
    model_id = np.random.randint(1000000, 10000000 - 1)
    # Saving the model's state dictionary
    if not os.path.exists("../saved_models/" + dataset_name):
        os.makedirs("../saved_models/" + dataset_name)
    num = model.number_of_layers
    torch.save(deepcopy(model).cpu().state_dict(), f'../saved_models/{dataset_name}/model_for_vasc_3d_{num}l_{model_id}.pth')
    print(f"Model saved as model_for_vasc_3d_{num}l_{model_id}.pth")
    # Set the model to evaluation mode
    return model, positive_accuracy, negative_accuracy
