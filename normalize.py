import config
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


# path to the folder with already segmented data
path_folder = "./preparations/data/outdata/"

# get a list of paths to the files
dirs = os.listdir(path_folder)
path_files = [os.path.join(path_folder, x) for x in dirs]

# find borders for a stack of pictures
fl = 0
for x in path_files:
    img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    if fl == 0:
        bordLeft, bordRight, bordTop, bordBottom = findBorders(img, img.shape[0] - 1, 0, img.shape[1] - 1, 0)
        bordLeftTmp, bordRightTmp, bordTopTmp, bordBottomTmp = bordLeft, bordRight, bordTop, bordBottom
        fl += 1
    else:
        bordLeftTmp, bordRightTmp, bordTopTmp, bordBottomTmp = findBorders(img, bordLeft, bordRight, bordTop, bordBottom)

    bordLeft = min(bordLeft, bordLeftTmp)
    bordRight = max(bordRight, bordRightTmp)
    bordTop = min(bordTop, bordTopTmp)
    bordBottom = max(bordBottom, bordBottomTmp)

print(bordLeft, bordRight, bordTop, bordBottom)

horizontal = bordRight - bordLeft
vertical = bordBottom - bordTop

print("Initial shape: ({0}, {1})".format(vertical, horizontal))

horizontalExtra = horizontal % config.WIDTH
verticalExtra = vertical % config.HEIGHT

print("Extras : ({0}, {1})".format(verticalExtra, horizontalExtra))

bordLeft = int(bordLeft + horizontalExtra / 2)
bordRight = int(bordRight - horizontalExtra / 2)

bordTop = int(bordTop + verticalExtra / 2)
bordBottom = int(bordBottom - verticalExtra / 2)

horizontal = bordRight - bordLeft
vertical = bordBottom - bordTop

print("New shape: ({0}, {1})".format(vertical, horizontal))

def cut_data(in_path1, in_path2, out_path1, out_path2, borders):
    bordTop, bordLeft = borders
    ind = 10000000
    files_in1 = os.listdir(in_path1)
    files_in2 = os.listdir(in_path2)
    sizes = []

    vertical_split = vertical // config.HEIGHT
    horizontal_split = horizontal // config.WIDTH

    # Create a tqdm progress bar for iterating through the files
    for x in tqdm(files_in2, desc="Processing"):
        img_tmp = cv2.imread(os.path.join(in_path1, x), cv2.IMREAD_GRAYSCALE)
        for i in range(vertical_split):
            for j in range(horizontal_split):
                img_tmp1 = img_tmp[bordTop  + config.HEIGHT * i: bordTop  + config.HEIGHT * (i + 1), bordLeft + config.WIDTH * j : bordLeft + config.WIDTH * (j + 1)]
                if np.sum(img_tmp1) != 0:
                    img_tmp2 = cv2.imread(os.path.join(in_path2, x), cv2.IMREAD_GRAYSCALE)[bordTop  + config.HEIGHT * i: bordTop  + config.HEIGHT * (i + 1), bordLeft + config.WIDTH * j : bordLeft + config.WIDTH * (j + 1)]
                    sizes.append(np.mean(img_tmp1))
    
    med = np.median(sizes)
    for i in tqdm(range(len(sizes)), "Writting images"):
        if sizes[i] >= med:
            grid_size = vertical_split * horizontal_split
            pic_number = i // grid_size
            vertical_number = (i % grid_size) // horizontal_split
            horizontal_number = (i % grid_size) % horizontal_split
            img_tmp1 = cv2.imread(os.path.join(in_path1, files_in1[pic_number]), cv2.IMREAD_GRAYSCALE)[bordTop  + config.HEIGHT * vertical_number: bordTop  + config.HEIGHT * (vertical_number + 1), bordLeft + config.WIDTH * horizontal_number : bordLeft + config.WIDTH * (horizontal_number + 1)]
            img_tmp2 = cv2.imread(os.path.join(in_path2, files_in2[pic_number]), cv2.IMREAD_GRAYSCALE)[bordTop  + config.HEIGHT * vertical_number: bordTop  + config.HEIGHT * (vertical_number + 1), bordLeft + config.WIDTH * horizontal_number : bordLeft + config.WIDTH * (horizontal_number + 1)]
            cv2.imwrite(os.path.join(out_path1, "img_" + str(ind) + ".png"), img_tmp1)
            cv2.imwrite(os.path.join(out_path2, "img_" + str(ind) + ".png"), img_tmp2)
            ind += 1

    # for x in tqdm(files_in2, desc="Processing"):
    #     for i in range(vertical // config.HEIGHT):
    #         for j in range(horizontal // config.WIDTH):
    #             img_tmp1 = cv2.imread(os.path.join(in_path1, x), cv2.IMREAD_GRAYSCALE)[bordTop  + config.HEIGHT * i: bordTop  + config.HEIGHT * (i + 1), bordLeft + config.WIDTH * j : bordLeft + config.WIDTH * (j + 1)]
    #             if np.sum(img_tmp1) >= np.median(sizes):
    #                 img_tmp2 = cv2.imread(os.path.join(in_path2, x), cv2.IMREAD_GRAYSCALE)[bordTop  + config.HEIGHT * i: bordTop  + config.HEIGHT * (i + 1), bordLeft + config.WIDTH * j : bordLeft + config.WIDTH * (j + 1)]
    #                 cv2.imwrite(os.path.join(out_path1, "img_" + str(ind) + ".png"), img_tmp1)
    #                 cv2.imwrite(os.path.join(out_path2, "img_" + str(ind) + ".png"), img_tmp2)
    #                 ind += 1

path_folder1 = "./preparations/data/outdata/"
path_folder1_cut = "./dataset/outdata/"

path_folder2 = "./preparations/data/indata/"
path_folder2_cut = "./dataset/indata/"

cut_data(path_folder1, path_folder2, path_folder1_cut, path_folder2_cut, (bordTop, bordLeft))

