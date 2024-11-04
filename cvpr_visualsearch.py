import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
import cvpr_compare
import config

# Load all descriptors
ALLFEAT = []
ALLFILES = []
for filename in os.listdir(os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER)):
    if filename.endswith(config.DEFAULT_DESCR_FILE_EXT):
        img_path = os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER, filename)
        img_data = sio.loadmat(img_path)
        ALLFILES.append(img_path)
        ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array

# Convert ALLFEAT to a numpy array
ALLFEAT = np.array(ALLFEAT)

# Pick a random image as the query
NIMG = ALLFEAT.shape[0]
queryimg = randint(0, NIMG - 1)

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[queryimg]
for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cvpr_compare.cvpr_compare(query, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])

# Show the top 15 results
SHOW = 15
for i in range(SHOW):
    img_filename = os.path.basename(ALLFILES[dst[i][1]]).replace(config.DEFAULT_DESCR_FILE_EXT, config.DEFAULT_IMG_FILE_EXT)

    img = cv2.imread(os.path.join(config.DATASET_IMAGES_FOLDER, img_filename))
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    cv2.imshow(f"Result {i+1}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()