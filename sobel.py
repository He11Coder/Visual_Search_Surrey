import numpy as np

def sobelFilterRGB(img):
    sobel_kernel_x = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])

    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])
    
    height, width, channels = img.shape

    grad_magn = np.empty((height, width, channels), dtype=np.float64)
    grad_angle = np.empty((height, width, channels), dtype=np.float64)

    for k in range(channels):
        for i in range(1, height-1):
            for j in range(1, width-1):
                region_to_convolve = img[i-1:i+2, j-1:j+2, k]

                Gx = np.sum(region_to_convolve * sobel_kernel_x)
                Gy = np.sum(region_to_convolve * sobel_kernel_y)

                grad_magn[i, j, k] = np.sqrt(Gx**2 + Gy**2)
                grad_angle[i, j, k] = np.atan2(Gy, Gx)

    grad_magn = np.clip(grad_magn, 0, 255).astype(np.uint8)

    return grad_magn, grad_angle


def sobelFilter(grayscale_img):
    sobel_kernel_x = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])

    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])
    
    height, width = grayscale_img.shape

    grad_magn = np.empty((height, width), dtype=np.float64)
    grad_angle = np.empty((height, width), dtype=np.float64)

    for i in range(1, height-1):
        for j in range(1, width-1):
            region_to_convolve = grayscale_img[i-1:i+2, j-1:j+2]

            Gx = np.sum(region_to_convolve * sobel_kernel_x)
            Gy = np.sum(region_to_convolve * sobel_kernel_y)

            grad_magn[i, j] = np.sqrt(Gx**2 + Gy**2)
            grad_angle[i, j] = np.atan2(Gy, Gx)

    grad_magn = np.clip(grad_magn, 0, 255).astype(np.uint8)
    grad_angle += np.pi

    return grad_magn, grad_angle