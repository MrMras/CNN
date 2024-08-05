import numpy as np
import os
import torch
import config
import sys

from scipy.ndimage import gaussian_filter
from model_structures.models_together import UNet3D
from tqdm import tqdm
from copy import deepcopy as copy

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the PyTorch model
path_model = "./saved_models/LSM/model_for_vasc_3d_2l_8688754.pth"
name = path_model.split("/")[-2]
model = UNet3D(number_of_layers=2)

# Load the model state dict
state_dict = torch.load(path_model, map_location="cpu")

# Adapt the keys in the state_dict if necessary
new_state_dict = {}
for key in state_dict.keys():
    new_key = key.replace("enc_conv", "encoder").replace("dec_conv", "decoder").replace("final", "out_conv")
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict, strict=False)

# Move the model to the GPU if available
model.to(device)

# Set the model to evaluation mode
model.eval()

def get_prediction(img, margin):
    if margin != 0:
        raise ValueError
    with torch.no_grad():
        input_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
        output = model(input_tensor)
    output_np = output.cpu().numpy()
    return output_np

def calculate_binary_array(count_array, total_array):
    mask = total_array == 0
    probability_array = np.zeros_like(count_array)
    new_array = np.zeros_like(count_array)
    probability_array[~mask] = count_array[~mask] / total_array[~mask]
    new_array[~mask] = probability_array[~mask] >= 0.5
    return probability_array, new_array

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
        blur_array = [gaussian_filter(image_array[i], sigma=1) for i in range(image_array.shape[0])]
        image_init = np.expand_dims(blur_array, axis=0)
    else:
        blur_array = [gaussian_filter(image_array[i], sigma=1) for i in range(image_array.shape[0])]
        image_pff_array = [np.floor(image_array[i] / blur_array[i] * 255) for i in range(image_array.shape[0])]
        image_pff_array = (image_pff_array - np.min(image_pff_array)) / (np.max(image_pff_array) - np.min(image_pff_array))
        image_init = np.expand_dims(image_pff_array, axis=0)
else:
    image_init = np.expand_dims(image_array, axis=0)
image_init = (image_init - np.min(image_init)) / (np.max(image_init) - np.min(image_init))

shape = image_init.shape
print("Initial shape:", shape)

step_ratio = 2
step = int(config.HEIGHT / step_ratio)
size = config.HEIGHT
print(step, size)
margin = 0

count_array = np.zeros_like(image_init)
total_array = np.zeros_like(image_init)

index = 0
if not os.path.exists("./nrrd"):
    os.makedirs("./nrrd")

i_amount = np.ceil(shape[1] / config.NUM_PICS)
j_amount = np.ceil(shape[2] / step)
k_amount = np.ceil(shape[3] / step)

for i in tqdm(range(0, int(i_amount)), "Processing"):
    for j in range(0, int(j_amount)):
        for k in range(0, int(k_amount)):
            slices = calculate_slices(i, j, k, shape)
            slice_tmp = image_init[slices]
            array_tmp = get_prediction(slice_tmp, margin)[0]
            count_array[slices] = count_array[slices] + array_tmp
            total_array[slices] += 1

binary_array, probability_array = calculate_binary_array(count_array, total_array)

if not os.path.exists("./processed_npy"):
    os.makedirs("./processed_npy")
if not os.path.exists("./probability_npy"):
    os.makedirs("./probability_npy")

np.save(f"./processed_npy/{name}.npy", binary_array)
np.save(f"./probability_npy/{name}.npy", probability_array)