import numpy as np
import cv2

import config
import sobel

def extractAverageRGB(img):
    blue_channel_mean = np.mean(img[:, :, 0])
    green_channel_mean = np.mean(img[:, :, 1])
    red_channel_mean = np.mean(img[:, :, 2])

    return np.array([red_channel_mean, green_channel_mean, blue_channel_mean], dtype=np.float64)


def extractGlobalColourHist(img):
    Q = config.QUANTIZATION_LEVEL
    hist = np.zeros(shape=(Q, Q, Q), dtype=np.uint32)

    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            pixel = img[i][j]
            b = np.uint32(np.floor(Q*pixel[0]))
            g = np.uint32(np.floor(Q*pixel[1]))
            r = np.uint32(np.floor(Q*pixel[2]))
            hist[r][g][b] += 1

    flattened_hist = hist.ravel()

    return np.divide(flattened_hist, np.sum(flattened_hist))


def extractAverageRGBGridded(num_x_cells, num_y_cells, img):
    x_side_of_cell = img.shape[1] // num_x_cells
    y_side_of_cell = img.shape[0] // num_y_cells

    descr = np.empty(num_x_cells*num_y_cells*3, dtype=np.float64)

    index_counter = 0
    for i in range(num_y_cells):
        for j in range(num_x_cells):
            cell = img[i*y_side_of_cell : (i+1)*y_side_of_cell, j*x_side_of_cell : (j+1)*x_side_of_cell]
            averageRGB = extractAverageRGB(cell)

            descr[index_counter : index_counter+3] = averageRGB

            index_counter += 3
    
    return descr


def extractHOG(magnitude, angle):
    threshold = config.THRESHOLD_PERCENTILE*magnitude.max()
    strong_edges_mask = magnitude >= threshold

    hist = np.empty(config.ANGLE_QUANTIZATION_LEVEL, dtype=np.uint32)

    bin_width = 2*np.pi / config.ANGLE_QUANTIZATION_LEVEL
    for i in range(config.ANGLE_QUANTIZATION_LEVEL):
        lower_bound = i*bin_width
        upper_bound = (i+1)*bin_width

        lower_bound_mask = angle >= lower_bound
        upper_bound_mask = angle < upper_bound
        current_bin_mask = lower_bound_mask & upper_bound_mask & strong_edges_mask

        if i == config.ANGLE_QUANTIZATION_LEVEL - 1:
            current_bin_mask = lower_bound_mask & (angle <= upper_bound) & strong_edges_mask
        
        hist[i] = np.sum(current_bin_mask.astype(np.uint8))

    return np.divide(hist, np.sum(hist))


def extractColourHOGGridded(num_x_cells, num_y_cells, rgb_img):
    x_side_of_cell = rgb_img.shape[1] // num_x_cells
    y_side_of_cell = rgb_img.shape[0] // num_y_cells

    grayscale_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    magn, angle = sobel.sobelFilter(grayscale_image)

    descr = np.empty(num_x_cells*num_y_cells*(3 + config.ANGLE_QUANTIZATION_LEVEL), dtype=np.float64)

    rgb_img = rgb_img.astype(np.float64) / 256.0

    index_counter = 0
    for i in range(num_y_cells):
        for j in range(num_x_cells):
            cell = rgb_img[i*y_side_of_cell : (i+1)*y_side_of_cell, j*x_side_of_cell : (j+1)*x_side_of_cell]
            averageRGB = extractAverageRGB(cell)

            hog = extractHOG(magn[i*y_side_of_cell : (i+1)*y_side_of_cell, j*x_side_of_cell : (j+1)*x_side_of_cell],
                             angle[i*y_side_of_cell : (i+1)*y_side_of_cell, j*x_side_of_cell : (j+1)*x_side_of_cell])

            concatenated = np.empty(3 + config.ANGLE_QUANTIZATION_LEVEL, dtype=np.float64)
            concatenated[:config.ANGLE_QUANTIZATION_LEVEL] = hog
            concatenated[config.ANGLE_QUANTIZATION_LEVEL:] = averageRGB

            descr[index_counter : index_counter + (3 + config.ANGLE_QUANTIZATION_LEVEL)] = concatenated

            index_counter += (3 + config.ANGLE_QUANTIZATION_LEVEL)
    
    return descr

'''
v = np.array([[[27, 27, 28], [27, 14, 28]], [[14,  5,  4],[ 5,  6, 14]]])

tuple_matrix = np.apply_along_axis(lambda x: tuple(x), 2, v)

img = cv2.imread(os.path.join(config.DATASET_IMAGES_FOLDER, "1_1_s.bmp")).astype(np.float64) / 255.0


print(np.divide(v.ravel(), np.sum(v.ravel())))

print(np.uint32(np.floor(4*0.26)))'''