import numpy as np

def extractAverageRGB(img):
    red_channel_mean = np.mean(img[:, :, 0])
    green_channel_mean = np.mean(img[:, :, 1])
    blue_channel_mean = np.mean(img[:, :, 2])

    return np.array([red_channel_mean, green_channel_mean, blue_channel_mean])