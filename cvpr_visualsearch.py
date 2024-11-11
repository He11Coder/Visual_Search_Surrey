import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
import matplotlib.pyplot as plt
from sklearn import metrics

import cvpr_compare
import metrics as pr_metric
import config

QUERY_IMAGES_DESCRIPTORS = ['2_6_s.bmp', '4_11_s.bmp', '13_10_s.bmp', '9_23_s.bmp', '11_5_s.bmp']

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
query_img_index = ALLFILES.index(QUERY_IMAGES_DESCRIPTORS[0])

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[query_img_index]
for i in range(config.NUM_OF_IMAGES):
    candidate = ALLFEAT[i]
    distance = cvpr_compare.L2Compare(query, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])
dst.pop(0)

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
'''SHOW = 20

for i in range(SHOW):
    img_filename = os.path.basename(ALLFILES[dst[i][1]]).replace(config.DEFAULT_DESCR_FILE_EXT, config.DEFAULT_IMG_FILE_EXT)

    img = cv2.imread(os.path.join(config.DATASET_IMAGES_FOLDER, img_filename))
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))



    cv2.imshow(f"Result {i+1}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()'''