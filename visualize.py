import torch
import os
import random
import cv2
import numpy as np

# Load the PyTorch model
model = torch.load("model_for_vasc.pth", map_location=torch.device('cpu'))  # Use 'cpu' if you don't have a GPU

# Set the model to evaluation mode
model.eval()

# Directory paths for input and output data
input_dir = "./dataset/indata"
output_dir = "./dataset/outdata"

# Get a list of image files in the input directory
image_files = os.listdir(input_dir)

# Choose a random pair of input and output files
random_image_file = random.choice(image_files)

# Construct the full file paths
input_image_path = os.path.join(input_dir, random_image_file)
output_label_path = os.path.join(output_dir, random_image_file)

# Check if the corresponding label file exists
if not os.path.exists(output_label_path):
    print(f"Label file for {random_image_file} not found.")
else:
    # Load the input image using OpenCV in grayscale mode
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Load the label image using OpenCV
    label_image = cv2.imread(output_label_path)

    # Perform inference using the model
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_image).unsqueeze(0).float()
        output = model(input_tensor)

    # Convert the output tensor to a NumPy array
    output_np = output.numpy()

    # Convert the output prediction to binary format and multiply by 255
    binary_output = (output_np < 0.5).astype(np.uint8) * 255

    # Display the input image, label, and binary prediction using OpenCV
    cv2.imshow("Input Image", input_image)
    cv2.imshow("Label Image", label_image)
    cv2.imshow("Model Prediction (Binary)", binary_output)

    # Wait for a key press and then close the OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
