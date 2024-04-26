import torch
import numpy as np

def get_stats(model, X_test, Y_test, device):
    # Set the model to evaluation mode
    model.eval()
    
    # Convert X_test to a tensor and move it to the appropriate device
    X_test_tensor = torch.Tensor(X_test).unsqueeze(1).to(device)

    # Initialize an empty list to store the predictions
    preds_test = []

    # Perform forward pass without gradient computation
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds_test.append(outputs.cpu().numpy())

    # Convert predictions to binary masks
    preds_test_binary = (np.concatenate(preds_test) > 0.5).astype(np.uint8)

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
    print("False Negatives (FN):", FN, end="\n\n")

    # Calculate other parameters
    print("Positive's accuracy:", accuracy(TP, FP, FN))
    print("Negative's accuracy:", accuracy(TN, FN, FP), end="\n\n")

    print("Positive's recall:", recall(TP, FP, FN))
    print("Negative's recaLL:", recall(TN, FN, FP), end="\n\n")

    print("Positive's precision:", precision(TP, FP, FN))
    print("Negative's precision:", precision(TN, FN, TP), end="\n\n")

    print("Positive's F1 score:", f1_score(TP, FP, FN))
    print("Negative's F1 score:", f1_score(TN, FN, TP), end="\n\n")

def accuracy(TP, FP, TN):
    return TP / (TP + TN + FP)

def precision(TP, FP, FN):
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(TP, FP, FN):
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(TP, FP, FN):
    return (2* TP) / (2 * TP + FP + FN)
