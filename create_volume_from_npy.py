import numpy as np
import cv2
import os
import config

from tqdm import tqdm

path_in = "./unprocessed_data/KESM/whole_volume_kesm.npy"
path_out = "./unprocessed_data/KESM/ground_truth_kesm.npy"

volume_in = np.load(path_in)
volume_out = np.load(path_out)

dataset_name = path_in.split("/")[-2]
print("Dataset name:", dataset_name)

def find_bound_indices(arr):
    val_max = np.max(arr)
    # Find indices where values are 255 along each axis
    x_indices = np.any(arr == val_max, axis=(1, 2))
    y_indices = np.any(arr == val_max, axis=(0, 2))
    z_indices = np.any(arr == val_max, axis=(0, 1))
    
    # Find the minimum and maximum indices that contain at least one 255 along each axis
    x_min, x_max = np.where(x_indices)[0][[0, -1]]
    y_min, y_max = np.where(y_indices)[0][[0, -1]]
    z_min, z_max = np.where(z_indices)[0][[0, -1]]
    
    return (x_min, x_max), (y_min, y_max), (z_min, z_max)

# Cut out empty regions
non_zero_indexes = find_bound_indices(volume_out)
print("Non-zero indexes:", non_zero_indexes)
volume_in = volume_in[
    non_zero_indexes[0][0]:non_zero_indexes[0][1]+1,
    non_zero_indexes[1][0]:non_zero_indexes[1][1]+1,
    non_zero_indexes[2][0]:non_zero_indexes[2][1]+1
    ]

volume_out = volume_out[
    non_zero_indexes[0][0]:non_zero_indexes[0][1]+1,
    non_zero_indexes[1][0]:non_zero_indexes[1][1]+1,
    non_zero_indexes[2][0]:non_zero_indexes[2][1]+1
    ]
print("Cut out empty regions, shape:", volume_in.shape)

# Match the size of array to CNN
NUMBER_OF_PICTURES_MOD = config.NUM_PICS
CNN_SIZE_MOD = config.HEIGHT

volume_in = volume_in[
    (non_zero_indexes[0][1]-non_zero_indexes[0][0]+1) % NUMBER_OF_PICTURES_MOD:,
    (non_zero_indexes[1][1]-non_zero_indexes[1][0]+1) % CNN_SIZE_MOD:,
    (non_zero_indexes[2][1]-non_zero_indexes[2][0]+1) % CNN_SIZE_MOD:
    ]

volume_out = volume_out[
    (non_zero_indexes[0][1]-non_zero_indexes[0][0]+1) % NUMBER_OF_PICTURES_MOD:,
    (non_zero_indexes[1][1]-non_zero_indexes[1][0]+1) % CNN_SIZE_MOD:,
    (non_zero_indexes[2][1]-non_zero_indexes[2][0]+1) % CNN_SIZE_MOD:
    ]
print("Matched the size, shape:", volume_in.shape)

array_flatten_in = []
array_flatten_out = []
for i in range(volume_in.shape[0] // config.NUM_PICS):
    for j in range(volume_in.shape[1] // config.HEIGHT):
        for k in range(volume_in.shape[2] // config.WIDTH):
            array_flatten_in.append(volume_in[i * config.NUM_PICS:(i + 1) * config.NUM_PICS, j * config.HEIGHT:(j + 1) * config.HEIGHT, k * config.WIDTH:(k + 1) * config.WIDTH])
            array_flatten_out.append(volume_out[i * config.NUM_PICS:(i + 1) * config.NUM_PICS, j * config.HEIGHT:(j + 1) * config.HEIGHT, k * config.WIDTH:(k + 1) * config.WIDTH])

array_flatten_in = np.array(array_flatten_in)
array_flatten_out = np.array(array_flatten_out)

print(array_flatten_in.shape)
print(array_flatten_out.shape)

# Calculate the sum along the last three axes of arr_out
arr_out_sum = np.sum(array_flatten_out, axis=(1, 2, 3))

# Find the indices where the sum is non-zero
non_zero_indices = np.nonzero(arr_out_sum)

# Extract only the elements where arr_out_sum is non-zero
arr_in_nonzero = array_flatten_in[non_zero_indices]
arr_out_nonzero = array_flatten_out[non_zero_indices]

# Check the shapes of the non-zero arrays
print("Shape of arr_in_nonzero:", arr_in_nonzero.shape)
print("Shape of arr_out_nonzero:", arr_out_nonzero.shape)

# Calculate the average along the first dimension
average_array = np.mean(arr_out_nonzero, axis=(1, 2, 3))

# Check the shape of the average_array
print("Shape of average_array:", average_array.shape)

# Compute the median of the average_array
median_value = np.median(average_array)
mean_value = np.mean(average_array)
print("Median value:", median_value)
print("Mean value:", mean_value)

# Create a boolean mask based on the condition (average more than median)
mask = average_array < max(mean_value, median_value)

# Check the shape of the mask
print("Shape of mask:", mask.shape)

# Remove elements along the first dimension where the mask is True
filtered_arr_in = arr_in_nonzero[~mask, :, :, :]
filtered_arr_out = arr_out_nonzero[~mask, :, :, :]

# Check the shapes of the filtered arrays
print("Shape of filtered arr_in:", filtered_arr_in.shape)
print("Shape of filtered arr_out:", filtered_arr_out.shape)

# Create folder ./data/micro_ct, if it doesn't exist
if not os.path.exists("./data/" + dataset_name):
    os.makedirs("./data/" + dataset_name)

np.save(f"./data/{dataset_name}/volume_input.npy", filtered_arr_in)
np.save(f"./data/{dataset_name}volume_ground_truth.npy", filtered_arr_out)