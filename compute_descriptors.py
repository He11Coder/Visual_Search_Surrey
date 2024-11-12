import os
import numpy as np
import cv2
import scipy.io as sio
import descriptor
import config

if not os.path.exists(os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER)):
    os.makedirs(os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER))

print("Start processing image files...")

for filename in os.listdir(config.DATASET_IMAGES_FOLDER):
    if filename.endswith(config.DEFAULT_IMG_FILE_EXT):
        img_path = os.path.join(config.DATASET_IMAGES_FOLDER, filename)
        img = cv2.imread(img_path)
        fout = os.path.join(config.DESCRIPTOR_FOLDER, config.DESCRIPTOR_SUBFOLDER, filename.replace(config.DEFAULT_IMG_FILE_EXT, config.DEFAULT_DESCR_FILE_EXT))

        if img.shape[0] > img.shape[1]:
            img = cv2.resize(img, (240, 320))
        else:
            img = cv2.resize(img, (320, 240))

        F = descriptor.extractColourHOGGridded(8, 8, img)
        sio.savemat(fout, {'F': F})

print("Description extraction finished")