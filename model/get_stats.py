import torch
import numpy as np

def get_stats(model, X_test_tensor, Y_test):
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize an empty list to store the predictions
    preds_test = []
    
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
