import numpy as np
import os
import torch
import config
import cv2

from model_structures.models_together import UNet3D
from tqdm import tqdm
from copy import deepcopy as copy

# Load the PyTorch model
path_model = "./saved_models/KESM/model_for_vasc_3d_1l_2247062.pth"
num = path_model.split("_")[-2][0]
name = path_model.split("/")[-2]
model = UNet3D(int(num))
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
    new_array = np.zeros_like(count_array)
    
    # Calculate binary_array where total_array is not zero
    new_array[~mask] = (count_array[~mask] / total_array[~mask]) >= 0.5
    
    return new_array

# Create a nrrd file from a prediction based on the input directory
image_array = np.load("./unprocessed_data/KESM/volume_input.npy")
# Apply Pseudo flat field correction
blur_array = [cv2.GaussianBlur(image_array[i], (127, 127), 0) for i in range(image_array.shape[0])]

image_pff_array = [cv2.divide(image_array[i], blur_array[i], scale=255) for i in range(image_array.shape[0])]

# normalize
image_pff_array = (image_pff_array - np.min(image_pff_array)) / (np.max(image_pff_array) - np.min(image_pff_array))

image_init = np.expand_dims(image_pff_array, axis=0)
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
            slice_tmp = image_init[:,
                                   min(i * config.NUM_PICS, shape[1] - config.NUM_PICS) : min((i + 1) * config.NUM_PICS, shape[1]),
                                   min(j * step, shape[2] - config.HEIGHT) : min(j * step + config.HEIGHT, shape[2]),
                                   min(k * step, shape[3] - config.WIDTH) : min(k * step + config.WIDTH, shape[3])]


            array_tmp = get_prediction(slice_tmp, margin)[0]
            # cv2.imshow(f"img", cv2.resize(slice_tmp[0][0], (512, 512)))
            # cv2.imshow("pred", cv2.resize(array_tmp[0][0], (512, 512)))
            # cv2.waitKey(0)

            count_array[:,
                        min(i * config.NUM_PICS, shape[1] - config.NUM_PICS) : min((i + 1) * config.NUM_PICS, shape[1]),
                        min(j * step, shape[2] - config.HEIGHT) : min(j * step + config.HEIGHT, shape[2]),
                        min(k * step, shape[3] - config.WIDTH) : min(k * step + config.WIDTH, shape[3])] = array_tmp
            total_array[:,
                        min(i * config.NUM_PICS, shape[1] - config.NUM_PICS) : min((i + 1) * config.NUM_PICS, shape[1]),
                        min(j * step, shape[2] - config.HEIGHT) : min(j * step + config.HEIGHT, shape[2]),
                        min(k * step, shape[3] - config.WIDTH) : min(k * step + config.WIDTH, shape[3])] = 1

binary_array = calculate_binary_array(count_array, total_array)[0] 
# Save the binary array as a .npy file
if not os.path.exists("./processed_npy"):
    os.makedirs("./processed_npy")

np.save(f"./processed_npy/{name}.npy", binary_array)