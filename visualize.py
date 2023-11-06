import cv2
import numpy as np
import os
import random
import torch
import config

from model_structure import UNet
from tqdm import tqdm

# Load the PyTorch model
model = UNet()
model.load_state_dict(torch.load("./model_for_vasc.pth", map_location="cpu"))

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
    label_image = cv2.imread(output_label_path, cv2.IMREAD_GRAYSCALE)

    # Perform inference using the model
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_image / 255).unsqueeze(0).float()
        output = model(input_tensor)

    # Convert the output tensor to a NumPy array
    output_np = output.numpy()

    # Convert the output prediction to binary format and multiply by 255
    print((np.unique(output_np).size))
    binary_output = (output_np >= 0.1).astype(np.uint8)[0] * 255

    # Display the input image, label, and binary prediction using OpenCV
    # cv2.imshow("Input Image", cv2.resize(input_image, (512, 512)))
    # cv2.imshow("Label Image", cv2.resize(label_image, (512, 512)))
    # cv2.imshow("Model Prediction (Binary)", cv2.resize(binary_output, (512, 512)))
    # cv2.imshow("Correctness Image", cv2.resize((np.multiply(binary_output, label_image) + np.multiply((255 - binary_output), (255 - label_image))) * 255, (512, 512)))
    
    print(np.sum(np.multiply(binary_output, label_image) + np.multiply((255 - binary_output), (255 - label_image))) / 64 / 64)

    

    # Wait for a key press and then close the OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arr1 = []
arr2 = []
for image_file in tqdm(image_files, "Processing"):
    # Construct the full file paths
    input_image_path = os.path.join(input_dir, image_file)
    output_label_path = os.path.join(output_dir, image_file)

    # Load the input image using OpenCV in grayscale mode
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Load the label image using OpenCV
    label_image = cv2.imread(output_label_path, cv2.IMREAD_GRAYSCALE)

    # Perform inference using the model
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_image).unsqueeze(0).float()
        output = model(input_tensor)

    # Convert the output tensor to a NumPy array
    output_np = output.numpy()

    # Convert the output prediction to binary format and multiply by 255
    binary_output = (output_np >= 0.5).astype(np.uint8)[0] * 255

    arr1.append(np.sum(np.multiply(binary_output, label_image)) / 64 / 64)
    arr2.append(np.sum(np.multiply((255 - binary_output), (255 - label_image))) / 64 / 64)
    
print(np.mean(arr1))
print(np.mean(arr2))