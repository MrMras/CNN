import numpy as np
import os
import torch
import config
import cv2
import sys

from model_structures.models_together import UNet3D
from tqdm import tqdm
from copy import deepcopy as copy

# Load the PyTorch model
path_model = "./saved_models/LSM/model_for_vasc_3d_2l_2464848.pth"
name = path_model.split("/")[-2]
model = UNet3D(number_of_layers=2)
model.load_state_dict(torch.load(path_model, map_location="cpu"))

# Set the model to evaluation mode
model.eval()

# Directory paths for input and output data
# input_dir = "./preparations/data/indata"
# output_dir = "./preparations/data/outdata"

# Get a list of image files in the input directory
# image_files = os.listdir(input_dir)

def get_prediction(img, margin):
    if margin != 0:
        raise ValueError
    with torch.no_grad():
        input_tensor = torch.from_numpy(img).unsqueeze(0).float()
        output = model(input_tensor)
    output_np = output.numpy()
    return output_np

def calculate_binary_array(count_array, total_array):
    # Avoid division by zero by setting a mask for total_array equal to zero
    mask = total_array == 0
    
    # Initialize binary_array with all zeros
    probabilty_array = np.zeros_like(count_array)
    new_array = np.zeros_like(count_array)
    
    # Calculate binary_array where total_array is not zero
    probabilty_array[~mask] = count_array[~mask] / total_array[~mask]
    new_array[~mask] = probabilty_array[~mask] >= 0.5
    
    return probabilty_array, new_array

def calculate_slices(i, j, k, shape):
    slices = (
        slice(None),
        slice(min(i * config.NUM_PICS, shape[1] - config.NUM_PICS), min((i + 1) * config.NUM_PICS, shape[1])),
        slice(min(j * step, shape[2] - config.HEIGHT), min(j * step + config.HEIGHT, shape[2])),
        slice(min(k * step, shape[3] - config.WIDTH), min(k * step + config.WIDTH, shape[3]))
    )
    return slices

# Create a nrrd file from a prediction based on the input directory
image_array = np.load("./unprocessed_data/LSM/volume_input.npy")
if len(sys.argv) > 1:
    if sys.argv[1] == "0":
        blur_array = [cv2.GaussianBlur(image_array[i], (5, 5), 0) for i in range(image_array.shape[0])]
        image_init = np.expand_dims(blur_array, axis=0)
    else:
        # Apply Pseudo flat field correction
        blur_array = [cv2.GaussianBlur(image_array[i], (127, 127), 0) for i in range(image_array.shape[0])]

        image_pff_array = [cv2.divide(image_array[i], blur_array[i], scale=255) for i in range(image_array.shape[0])]

        # normalize
        image_pff_array = (image_pff_array - np.min(image_pff_array)) / (np.max(image_pff_array) - np.min(image_pff_array))

        image_init = np.expand_dims(image_pff_array, axis=0)
else:
    image_init = np.expand_dims(image_array, axis=0)
# normalize
image_init = (image_init - np.min(image_init)) / (np.max(image_init) - np.min(image_init))

# Print the shape of the input and output arrays
shape = image_init.shape
print("Initial shape:", shape)

# Parameters
step_ratio = 2
step = int(config.HEIGHT / step_ratio)
size = config.HEIGHT
print(step, size)
margin = 0

# Create 0 copies of inputs
count_array = np.zeros_like(image_init)
total_array = np.zeros_like(image_init)

index = 0
if not os.path.exists("./nrrd"):
    os.makedirs("./nrrd")

# amount
i_amount = np.ceil(shape[1] / config.NUM_PICS)
j_amount = np.ceil(shape[2] / step)
k_amount = np.ceil(shape[3] / step)

for i in tqdm(range(0, int(i_amount)), "Processing"):
    for j in range(0, int(j_amount)):
        for k in range(0, int(k_amount)):
            # a b c based on i j k
            slices = calculate_slices(i, j, k, shape)
            slice_tmp = image_init[slices]
            array_tmp = get_prediction(slice_tmp, margin)[0]
            count_array[slices] = count_array[slices] + array_tmp
            total_array[slices] += 1

cv2.destroyAllWindows()

# Calculate the binary array
binary_array, probability_array = calculate_binary_array(count_array, total_array)[0]

# Save the binary array as a .npy file
if not os.path.exists("./processed_npy"):
    os.makedirs("./processed_npy")
if not os.path.exists("./probability_npy"):
    os.makedirs("./probability_npy")

np.save(f"./processed_npy/{name}.npy", binary_array)
np.save(f"./probability_npy/{name}.npy", probability_array)