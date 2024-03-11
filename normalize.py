import config
import concurrent.futures
import cv2
import numpy as np
import os
from tqdm import tqdm

def findBordersHorizontal(img, border, side):
    if side == "l2r":
        for i in range(border):
            if np.sum(img[:,i]) > 0:
                return i
    elif side == "r2l":
        for i in range(img.shape[1] - 1, border, -1):
            if np.sum(img[:,i]) > 0:
                return i
    else:
        raise ValueError
    return border

def findBordersVertical(img, border, side):
    if side == "t2b":
        for i in range(border):
            if np.sum(img[i]) > 0:
                return i
    elif side == "b2t":
        for i in range(img.shape[0] - 1, border, -1):
            if np.sum(img[i]) > 0:
                return i
    else:
        raise ValueError
    return border


def findBorders(img, bordLeft, bordRight, bordTop, bordBottom):
    bordLeft = findBordersHorizontal(img, bordLeft, "l2r")
    bordRight = findBordersHorizontal(img, bordRight, "r2l")
    bordTop = findBordersVertical(img, bordTop, "t2b")
    bordBottom = findBordersVertical(img, bordBottom, "b2t")
    return bordLeft, bordRight, bordTop, bordBottom


# Function to find borders for a single image
def find_borders(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return findBorders(img, img.shape[0] - 1, 0, img.shape[1] - 1, 0)

# Function to merge borders from multiple images
def merge_borders(borders):
    bordLeft, bordRight, bordTop, bordBottom = borders[0]
    for b in borders[1:]:
        bordLeftTmp, bordRightTmp, bordTopTmp, bordBottomTmp = b
        bordLeft = min(bordLeft, bordLeftTmp)
        bordRight = max(bordRight, bordRightTmp)
        bordTop = min(bordTop, bordTopTmp)
        bordBottom = max(bordBottom, bordBottomTmp)
    return bordLeft, bordRight, bordTop, bordBottom

# # path to the folder with already segmented data
# path_folder = "./preparations/data/outdata/"

# # get a list of paths to the files
# dirs = os.listdir(path_folder)
# path_files = [os.path.join(path_folder, x) for x in dirs]

# # Number of threads to use
# num_threads = min(len(path_files), 4)  # Adjust the number of threads as needed

# borders = []

# # Use ThreadPoolExecutor to parallelize the finding of borders
# with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#     # Submit tasks for each image to find borders concurrently
#     futures = [executor.submit(find_borders, path) for path in path_files]

#     # Collect results
#     for future in tqdm(concurrent.futures.as_completed(futures), "Finding borders"):
#         borders.append(future.result())

# # Merge borders from multiple images
# bordLeft, bordRight, bordTop, bordBottom = merge_borders(borders)

# print(bordLeft, bordRight, bordTop, bordBottom)

# horizontal = bordRight - bordLeft
# vertical = bordBottom - bordTop

# print("Initial shape: ({0}, {1})".format(vertical, horizontal))

# horizontalExtra = horizontal % config.WIDTH
# verticalExtra = vertical % config.HEIGHT

# print("Extras : ({0}, {1})".format(verticalExtra, horizontalExtra))

# bordLeft = int(bordLeft + horizontalExtra / 2)
# bordRight = int(bordRight - horizontalExtra / 2)

# bordTop = int(bordTop + verticalExtra / 2)
# bordBottom = int(bordBottom - verticalExtra / 2)

# horizontal = bordRight - bordLeft
# vertical = bordBottom - bordTop

# print("New shape: ({0}, {1})".format(vertical, horizontal))

vertical = 384
horizontal = 384
bordTop = 0
bordLeft = 0

def cut_data(in_path1, in_path2, out_path1, out_path2, borders):
    # Extracting border coordinates
    bordTop, bordLeft = borders
    # Starting index for saving images
    ind = 1000000
    # List of files in input paths
    files_in1 = os.listdir(in_path1)
    files_in2 = os.listdir(in_path2)
    # List to store mean intensities of processed images
    sizes = np.array([])

    # Calculating the number of splits vertically and horizontally
    vertical_split = vertical // config.HEIGHT
    horizontal_split = horizontal // config.WIDTH
         
    # Calculating grid size
    grid_size = vertical_split * horizontal_split
    
    for i in tqdm(range(vertical_split * horizontal_split), desc="Processing"):
        # Calculate the grid indices
        j = i % horizontal_split
        k = i // horizontal_split

        # Iterate through the image files in batches
        for i in range(0, len(files_in2) - len(files_in2) % config.NUM_PICS, config.NUM_PICS):
            # Initialize a list to store intensities for each image in the group
            group_intensities = np.array([])

            # Iterate through the images in the current group
            for l, x in enumerate(files_in2[i:i+config.NUM_PICS]):
                # Read and extract a region of interest from the image
                img_tmp1 = cv2.imread(os.path.join(in_path1, x), cv2.IMREAD_GRAYSCALE)
                img_tmp1 = img_tmp1[bordTop + config.HEIGHT * k: bordTop + config.HEIGHT * (k + 1), bordLeft + config.WIDTH * j: bordLeft + config.WIDTH * (j + 1)]
                # Calculate and store the mean intensity of the current image
                group_intensities = np.append(group_intensities, np.mean(img_tmp1))

            # Calculate the average intensity for the current group
            average_intensity = np.mean(group_intensities)

            # Store the average intensity in the 'sizes' list
            sizes = np.append(sizes, average_intensity)

    # Calculating the median of the mean intensities
    print(sizes.dtype)
    non_zero_values = sizes[sizes != 0]
    print("Uniques:", np.unique(sizes))
    med = np.median(non_zero_values)
    print("Median", med)
    mean = np.mean(non_zero_values)
    print("Mean", mean)
    # print(f"length - {len(files_in1)}")
    # print(f"length of sizes - {len(sizes)}")
    # print(f"grid_size - {grid_size}")
    # Iterating through each set of images and writing selected ones
    for i in tqdm(range(len(sizes)), "Writting images"):
        # Checking if the mean intensity of the set is greater than or equal to the median
        if sizes[i] >= med:
            # Extracting indices for the selected set of images
            pic_number = [(i * config.NUM_PICS + j) % (len(files_in1) -  len(files_in1) % config.NUM_PICS) for j in range(config.NUM_PICS)]
            horizontal_number = i // ((len(files_in1) -  len(files_in1) % config.NUM_PICS) // config.NUM_PICS) % horizontal_split
            vertical_number = i // ((len(files_in1) -  len(files_in1) % config.NUM_PICS) // config.NUM_PICS) // horizontal_split

            # Iterating through images in the selected set and writing them
            for k in range(config.NUM_PICS):
                
                # print(f"i - {i}\nk - {k}\npic_number[k] - {pic_number[k]}")
                img_tmp1 = cv2.imread(os.path.join(in_path1, files_in1[pic_number[k]]), cv2.IMREAD_GRAYSCALE)
                img_tmp2 = cv2.imread(os.path.join(in_path2, files_in2[pic_number[k]]), cv2.IMREAD_GRAYSCALE)

                img_tmp1 = img_tmp1[bordTop  + config.HEIGHT * vertical_number: bordTop  + config.HEIGHT * (vertical_number + 1), bordLeft + config.WIDTH * horizontal_number : bordLeft + config.WIDTH * (horizontal_number + 1)]
                img_tmp2 = img_tmp2[bordTop  + config.HEIGHT * vertical_number: bordTop  + config.HEIGHT * (vertical_number + 1), bordLeft + config.WIDTH * horizontal_number : bordLeft + config.WIDTH * (horizontal_number + 1)]
                cv2.imwrite(os.path.join(out_path1, f"img_{ind}_{k}.png"), img_tmp1)
                cv2.imwrite(os.path.join(out_path2, f"img_{ind}_{k}.png"), img_tmp2)
            ind += 1


def cut_data_2(data_in, data_out, out_path1, out_path2, borders):
    # Extracting border coordinates
    bordTop, bordLeft = borders
    # Starting index for saving images
    ind = 1000000
    # List of files in input paths
    # List to store mean intensities of processed images
    sizes = np.array([])

    # Calculating the number of splits vertically and horizontally
    vertical_split = vertical // config.HEIGHT
    horizontal_split = horizontal // config.WIDTH
         
    # Calculating grid size
    grid_size = vertical_split * horizontal_split
    shape = data_in.shape

    for i in tqdm(range(vertical_split * horizontal_split), desc="Processing"):
        # Calculate the grid indices
        j = i % horizontal_split
        k = i // horizontal_split

        # Iterate through the image files in batches
        for i in range(0, shape[2] - shape[2] % config.NUM_PICS, config.NUM_PICS):
            # Initialize a list to store intensities for each image in the group
            group_intensities = np.array([])

            # Iterate through the images in the current group
            for i in range(shape[2]):
                # Read and extract a region of interest from the image
                img_tmp1 = data_out[bordTop + config.HEIGHT * k: bordTop + config.HEIGHT * (k + 1), bordLeft + config.WIDTH * j: bordLeft + config.WIDTH * (j + 1), i]
                # Calculate and store the mean intensity of the current image
                group_intensities = np.append(group_intensities, np.mean(img_tmp1))

            # Calculate the average intensity for the current group
            average_intensity = np.mean(group_intensities)

            # Store the average intensity in the 'sizes' list
            sizes = np.append(sizes, average_intensity)

    # Calculating the median of the mean intensities
    print(sizes.dtype)
    non_zero_values = sizes[sizes != 0]
    print("Uniques:", np.unique(sizes))
    med = np.median(non_zero_values)
    print("Median", med)
    mean = np.mean(non_zero_values)
    print("Mean", mean)
    # print(f"length - {len(files_in1)}")
    # print(f"length of sizes - {len(sizes)}")
    # print(f"grid_size - {grid_size}")
    # Iterating through each set of images and writing selected ones
    for i in tqdm(range(len(sizes)), "Writting images"):
        # Checking if the mean intensity of the set is greater than or equal to the median
        if sizes[i] >= med:
            # Extracting indices for the selected set of images
            pic_number = [(i * config.NUM_PICS + j) % (shape[2] - shape[2] % config.NUM_PICS) for j in range(config.NUM_PICS)]
            horizontal_number = i // ((shape[2] - shape[2] % config.NUM_PICS) // config.NUM_PICS) % horizontal_split
            vertical_number = i // ((shape[2] - shape[2] % config.NUM_PICS) // config.NUM_PICS) // horizontal_split

            # Iterating through images in the selected set and writing them
            for k in range(config.NUM_PICS):
                
                # print(f"i - {i}\nk - {k}\npic_number[k] - {pic_number[k]}")
                img_tmp1 = data_out[:, :, pic_number[k]]
                img_tmp2 = data_in[:, :, pic_number[k]]

                img_tmp1 = img_tmp1[bordTop  + config.HEIGHT * vertical_number: bordTop  + config.HEIGHT * (vertical_number + 1), bordLeft + config.WIDTH * horizontal_number : bordLeft + config.WIDTH * (horizontal_number + 1)]
                img_tmp2 = img_tmp2[bordTop  + config.HEIGHT * vertical_number: bordTop  + config.HEIGHT * (vertical_number + 1), bordLeft + config.WIDTH * horizontal_number : bordLeft + config.WIDTH * (horizontal_number + 1)]
                cv2.imwrite(os.path.join(out_path1, f"img_{ind}_{k}.png"), img_tmp1)
                cv2.imwrite(os.path.join(out_path2, f"img_{ind}_{k}.png"), img_tmp2)
            ind += 1

# # Define the directory path
# base_dir = './dataset'

# # Create the 'dataset' folder
# if not os.path.exists(base_dir):
#     os.makedirs(base_dir)

# # Create the 'indata' folder inside 'dataset'
# indata_dir = os.path.join(base_dir, 'indata')
# if not os.path.exists(indata_dir):
#     os.makedirs(indata_dir)

# # Create the 'outdata' folder inside 'dataset'
# outdata_dir = os.path.join(base_dir, 'outdata')
# if not os.path.exists(outdata_dir):
#     os.makedirs(outdata_dir)

# path_folder1 = "./preparations/data/outdata/"
# path_folder1_cut = "./dataset/outdata/"

# path_folder2 = "./preparations/data/indata/"
# path_folder2_cut = "./dataset/indata/" 

# cut_data(path_folder1, path_folder2, path_folder1_cut, path_folder2_cut, (bordTop, bordLeft))
            
data_in = np.load("./KESM/whole_volume_kesm.npy")
data_out = np.load("./KESM/ground_truth_kesm.npy")
base_dir = './dataset_kesm'
path_folder1_cut = "./dataset_kesm/outdata/"
path_folder2_cut = "./dataset_kesm/indata/"

# Create the 'dataset' folder
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Create the 'indata' folder inside 'dataset'
indata_dir = os.path.join(base_dir, 'indata')
if not os.path.exists(indata_dir):
    os.makedirs(indata_dir)

# Create the 'outdata' folder inside 'dataset'
outdata_dir = os.path.join(base_dir, 'outdata')
if not os.path.exists(outdata_dir):
    os.makedirs(outdata_dir)


cut_data_2(data_in, data_out, path_folder1_cut, path_folder2_cut, (bordTop, bordLeft))