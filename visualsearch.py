import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
import matplotlib.pyplot as plt
from sklearn import metrics

import compare
import metrics as pr_metric
import config

QUERY_IMAGES = ['2_6_s.bmp', '4_11_s.bmp', '13_10_s.bmp', '8_29_s.bmp']
QUERY_IMAGE = QUERY_IMAGES[0]

# Load all descriptors
ALLFEAT = []
ALLFILES = []
for filename in os.listdir(os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER)):
    if filename.endswith(config.DEFAULT_DESCR_FILE_EXT):
        desc_path = os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER, filename)
        desc_data = sio.loadmat(desc_path)
        ALLFILES.append(os.path.basename(desc_path).replace(config.DEFAULT_DESCR_FILE_EXT, config.DEFAULT_IMG_FILE_EXT))
        ALLFEAT.append(desc_data['F'][0])  # Assuming F is a 1D array

# Convert ALLFEAT to a numpy array
ALLFEAT = np.array(ALLFEAT)

# Pick a random image as the query
query_img_index = ALLFILES.index(QUERY_IMAGE)

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[query_img_index]
for i in range(config.NUM_OF_IMAGES):
    candidate = ALLFEAT[i]
    distance = compare.L1Compare(query, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])
dst.pop(0)

# Plot PR Curve and calculate PR-AUC
num_of_results = len(dst)

image_file_names = []
for i in range(num_of_results):
    image_file_names.append(ALLFILES[dst[i][1]])

pr_stat = pr_metric.computePR(ALLFILES[query_img_index], image_file_names)

pr_auc = metrics.auc(pr_stat['f0'], pr_stat['f1'])

plt.plot(pr_stat['f0'], pr_stat['f1'], color='orange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-AUC Curve | ' + f'PR_AUC = {pr_auc:.3f}')
plt.grid()

plt.show()

# Show the top 10 results
'''SHOW = 10

top_images = []
for i in range(SHOW):
    img_filename = os.path.basename(ALLFILES[dst[i][1]]).replace(config.DEFAULT_DESCR_FILE_EXT, config.DEFAULT_IMG_FILE_EXT)

    img = cv2.imread(os.path.join(config.DATASET_IMAGES_FOLDER, img_filename))
    img = cv2.resize(img, (120, 160))

    top_images.append(img)

query_img = cv2.imread(os.path.join(config.DATASET_IMAGES_FOLDER, QUERY_IMAGE))
query_img = cv2.resize(query_img, (240, 320))

horiz1 = np.concatenate((top_images[0], top_images[1], top_images[2], top_images[3], top_images[4]), axis = 1)
horiz2 = np.concatenate((top_images[5], top_images[6], top_images[7], top_images[8], top_images[9]), axis = 1)

result = np.concatenate((horiz1, horiz2), axis = 0)

query_result = np.concatenate((query_img, result), axis = 1)

cv2.imshow("Search result", query_result)
cv2.waitKey(0)

cv2.destroyAllWindows()'''