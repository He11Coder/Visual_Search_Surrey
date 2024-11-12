import os
import numpy as np

DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
DATASET_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, 'Images')
DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'ColourHOGGridded_64cells_16bins_0.5Thr'

DEFAULT_IMG_FILE_EXT = '.bmp'
DEFAULT_DESCR_FILE_EXT = '.mat'

NUMBER_OF_CLASSES = 20

NUM_OF_ELS_IN_CLASS = np.zeros(NUMBER_OF_CLASSES, dtype=np.int32)
for filename in os.listdir(DATASET_IMAGES_FOLDER):
    if filename.endswith(DEFAULT_IMG_FILE_EXT):
        class_number = int(filename.split('_')[0])
        NUM_OF_ELS_IN_CLASS[class_number-1] += 1

NUM_OF_IMAGES = sum(NUM_OF_ELS_IN_CLASS)

#Quantization level of Global Colour Histogram (how many bins are along each axis)
QUANTIZATION_LEVEL = 8

#Quantization level of Sobel Gradient Direction (how many bins in 2pi radians)
ANGLE_QUANTIZATION_LEVEL = 8
#Specifies the boundary from which we can consider Sobel Gradient magnitude as significant (strong edge)
THRESHOLD_PERCENTILE = 0.5