import numpy as np
import os
import torch
import config

from model_structures.model_structure_3d_1l import UNet3D
from tqdm import tqdm
from copy import deepcopy as copy

# Load the PyTorch model
model = UNet3D()
model.load_state_dict(torch.load("./saved_models/KESM/model_for_vasc_3d9803342.pth", map_location="cpu"))

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
    new_array = np.zeros_like(count_array)
    
    # Calculate binary_array where total_array is not zero
    new_array[~mask] = (count_array[~mask] / total_array[~mask]) >= 0.5
    
    return new_array

# Create a nrrd file from a prediction based on the input directory
image_array = np.load("./unprocessed_data/KESM/whole_volume_kesm.npy")

image_init = np.expand_dims(image_array, axis=0)
# Print the shape of the input and output arrays
shape = image_init.shape
print("Initial shape:", shape)

# Parameters
step_ratio = 1
step = int(step_ratio * config.HEIGHT)
margin = 0

# Create 0 copies of inputs
count_array = np.zeros_like(image_init)
total_array = np.zeros_like(image_init)

index = 0
if not os.path.exists("./nrrd"):
    os.makedirs("./nrrd")
nrrd_array = []
# Loop over the input images
for i in tqdm(range(0, shape[1] - shape[1] % 16, 16), "Processing"):
    for j in range(0, shape[2] - shape[2] % step , step):
        for k in range(0, shape[3] - shape[3] % step , step):
            i_mod = (16 - i % 16) % 16
            j_mod = (step - j % step) % step
            k_mod = (step - k % step) % step

            slice_tmp = image_init[:, i - i_mod : i + 16 - i_mod, j - j_mod : j + config.HEIGHT - j_mod, k - k_mod : k + config.WIDTH - k_mod]
            array_tmp = get_prediction(slice_tmp / 255, margin)[0]
            count_array[:, i - i_mod : i + 16 - i_mod, j - j_mod : j + config.HEIGHT - j_mod, k - k_mod : k + config.WIDTH - k_mod] = array_tmp
            total_array[:, i - i_mod : i + 16 - i_mod, j - j_mod : j + config.HEIGHT - j_mod, k - k_mod : k + config.WIDTH - k_mod] += 1
    
binary_array = calculate_binary_array(count_array, total_array)[0] * 255
# Save the binary array as a .npy file
if not os.path.exists("./processed_npy"):
    os.makedirs("./processed_npy")

np.save("./processed_npy/kesm.npy", binary_array)