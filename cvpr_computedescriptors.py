import os
import numpy as np
import cv2
import scipy.io as sio
import extractRandom
import config

if not os.path.exists(config.DESCRIPTOR_FOLDER):
    os.makedirs(os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER))

for filename in os.listdir(config.DATASET_IMAGES_FOLDER):
    if filename.endswith(config.DEFAULT_IMG_FILE_EXT):
        print(f"Processing file {filename}")
        img_path = os.path.join(config.DATASET_IMAGES_FOLDER, filename)
        img = cv2.imread(img_path).astype(np.float64) / 255.0 
        fout = os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER, filename.replace(config.DEFAULT_IMG_FILE_EXT, config.DEFAULT_DESCR_FILE_EXT))
        
        F = extractRandom.extractRandom(img)
        
        sio.savemat(fout, {'F': F})