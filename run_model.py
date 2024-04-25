import numpy as np
import os
import torch
import config

from model_structures.model_structure_3d_1l import UNet3D
from tqdm import tqdm
from copy import deepcopy as copy

# Load the PyTorch model
model = UNet3D()
model.load_state_dict(torch.load("./saved_models/KESM/model_for_vasc_3d3733964.pth", map_location="cpu"))

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
path = "./unprocessed_data/KESM/whole_volume_kesm.npy"
image_array = np.load(path)
name = path.split("/")[-2]

image_init = np.expand_dims(image_array, axis=0)
# Print the shape of the input and output arrays
shape = image_init.shape
print("Initial shape:", shape)

# Parameters
step_ratio = 1
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
nrrd_array = []

# amount
i_amount = int(np.ceil(shape[1] / 16))
j_amount = int(np.ceil(shape[2] / step))
k_amount = int(np.ceil(shape[3] / step))

for i in tqdm(range(0, i_amount), "Processing"):
    for j in range(0, j_amount):
        for k in range(0, k_amount):
            # a b c based on i j k
            
            if shape[1] - 16 * i < 16:
                a = shape[1] - 16
            else:
                a = 16 * i

            if shape[2] - step * j < config.HEIGHT:
                b = shape[2] - config.HEIGHT
            else:
                b = step * j

            if shape[3] - step * k < config.WIDTH:
                c = shape[3] - config.WIDTH
            else:
                c = step * k
            # print(c)
            # print(a, a + 16, "\n",b, b + config.HEIGHT, "\n", c, c + config.WIDTH)
            slice_tmp = image_init[:, a : a + 16, b : b + config.HEIGHT, c: c + config.WIDTH]
            array_tmp = get_prediction(slice_tmp / 255, margin)[0]

            count_array[:, a : a + 16, b : b + config.HEIGHT, c: c + config.WIDTH] = array_tmp
            total_array[:, a : a + 16, b : b + config.HEIGHT, c: c + config.WIDTH] += 1

binary_array = calculate_binary_array(count_array, total_array)[0] * 255
# Save the binary array as a .npy file
if not os.path.exists("./processed_npy"):
    os.makedirs("./processed_npy")

np.save(f"./processed_npy/{name}.npy", binary_array)