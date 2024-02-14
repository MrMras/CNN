import config
import torch

from copy import deepcopy
from tqdm import tqdm

def train(model, criterion, optimizer, train_loader, device, epochs, weights):
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
    return model