'''
import cv2
# Define the prefix
prefix = "/root/mount_data/xjh/UniTR/data/nuscenes/v1.0-trainval"

# Assuming input_dict["image_paths"] is a list of relative paths
input_dict = {
    "image_paths": [
        "samples/CAM_FRONT/n008-2018-08-29-16-04-13-0400__CAM_FRONT__1535574132662404.jpg",
        "samples/CAM_FRO NT/n009-2018-08-29-16-04-13-0400__CAM_FRONT__1535574132662405.jpg",
        # Add more paths as needed
    ]
}

# Prepend the prefix to each path
input_dict["image_paths"] = [f"{prefix}/{path}" for path in input_dict["image_paths"]]
img_path=input_dict["image_paths"]

for path in img_path:
    print(f"Reading image from: {img_path}")
    img = cv2.imread(path)  # Try to load the image
    if img is None:
        print(f"Failed to read image from {img_path}")
    else:
        print(f"Successfully loaded image from {img_path}")
'''
'''
import cv2

# Define the prefix
prefix = "/root/mount_data/xjh/UniTR/data/nuscenes/v1.0-trainval"

# Assuming input_dict["image_paths"] is a list of relative paths
input_dict = {
    "image_paths": [
        "samples/CAM_FRONT/n008-2018-08-29-16-04-13-0400__CAM_FRONT__1535574132662404.jpg",
        # Add more paths as needed
    ]
}

# Prepend the prefix to each path
input_dict["image_paths"] = [f"{prefix}/{path}" for path in input_dict["image_paths"]]
img_paths = input_dict["image_paths"]


# Iterate through each path and load the image

for path in img_paths:
    print(f"Reading image from: {path}")
    img = cv2.imread(path)  # Try to load the image
    if img is None:
        print(f"Failed to read image from {path}")
    else:
        print(f"Successfully loaded image from {path}")
'''

import numpy as np
import torch
pc=[[1,2,3],[1,2,3],[4,5,6]]
print(np.shape(pc))

