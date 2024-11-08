import os

DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
DATASET_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, 'Images')
DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'

DEFAULT_IMG_FILE_EXT = '.bmp'
DEFAULT_DESCR_FILE_EXT = '.mat'

#Quantization level of Global Colour Histogram (how many bins are along each axis)
QUANTIZATION_LEVEL = 4