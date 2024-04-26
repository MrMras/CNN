import numpy as np
import nrrd
import os
import cv2

data = nrrd.read("./nrrd/v2_or_v3_1.nrrd")[0]
data = np.moveaxis(np.array(data), [2, 1], [0, -2])

# save as npy
# create a folder
path = "./unprocessed_data/micro_ct_v4"
if not os.path.exists(path):
    os.makedirs(path)
old_data = np.load("./unprocessed_data/micro_ct/volume_input.npy")
cv2.imshow("name", data[0] * 255)
cv2.imshow("name2", old_data[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


print(old_data.shape)
print(data.shape)
np.save(f"{path}/volume_input.npy", old_data)
np.save(f"{path}/volume_ground_truth.npy", data)