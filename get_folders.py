import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import cv2
from functools import reduce

for j in range(4):
    img_tl = nib.load("./preparations/Segmentations(NIFTI)/Segmentation_Vol" + str(j + 1) + "_topLeft.nii")
    img_tr = nib.load("./preparations/Segmentations(NIFTI)/Segmentation_Vol" + str(j + 1) + "_topRight.nii")
    img_bl = nib.load("./preparations/Segmentations(NIFTI)/Segmentation_Vol" + str(j + 1) + "_bottomLeft.nii")
    img_br = nib.load("./preparations/Segmentations(NIFTI)/Segmentation_Vol" + str(j + 1) + "_bottomRight.nii")

    a_tl = np.flip(np.moveaxis(np.array(img_tl.dataobj), [2, 1], [0, -2]), 2)
    a_tr = np.flip(np.moveaxis(np.array(img_tr.dataobj), [2, 1], [0, -2]), 2)
    a_bl = np.flip(np.moveaxis(np.array(img_bl.dataobj), [2, 1], [0, -2]), 2)
    a_br = np.flip(np.moveaxis(np.array(img_br.dataobj), [2, 1], [0, -2]), 2)

    for i in range(a_tl.shape[0]):
        bottom_img = np.concatenate((a_bl[i], a_br[i]), axis=1) * 255
        top_img = np.concatenate((a_tl[i], a_tr[i]), axis=1) * 255
        final = np.concatenate((bottom_img, top_img), axis=0)
        cv2.imwrite("./data/outdata/img_" + str(1000000 + i + 335 * j) + ".png", np.flip(final, axis=0))


folder_path = "./New folder/"
dirs = os.listdir(folder_path)
i = 1000000
for x in dirs:
    os.rename(os.path.join(folder_path, x), os.path.join("./indata_tif/", "img_" + str(i) + ".tif"))
    i += 1

folder_path = "./indata_tif/"
dirs = os.listdir(folder_path)

for x in dirs:
    cv2.imwrite("./indata/" + x[:-3] + "png", cv2.imread(os.path.join(folder_path, x)))


# Specify the folder containing the pictures
folder_path = "./data/outdata/"

# Get the list of picture files
picture_files = os.listdir(folder_path)

# Check if there are pictures in the folder
if picture_files:
    # Get the first picture's path to extract width and height
    first_picture_path = os.path.join(folder_path, picture_files[0])

    # Read the first picture to get its dimensions
    first_picture = cv2.imread(first_picture_path)
    height, width, _ = first_picture.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('indata_video.mp4', fourcc, 10, (width, height))

    # Iterate through the pictures and add them to the video
    for picture_file in picture_files:
        picture_path = os.path.join(folder_path, picture_file)
        frame = cv2.imread(picture_path)
        output_video.write(frame)

    # Release the VideoWriter when done
    output_video.release()

else:
    print("No pictures found in the folder.")
