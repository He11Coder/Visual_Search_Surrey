import config

def getImgFilenameFromTuple(distance_descr_tuple):
    descr_filename = distance_descr_tuple[1]
    return descr_filename.replace(config.DEFAULT_DESCR_FILE_EXT, config.DEFAULT_IMG_FILE_EXT)
