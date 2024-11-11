import numpy as np

def L2Compare(F1, F2):
    square_diffs = np.square(np.subtract(F1, F2))

    return np.sqrt(np.sum(square_diffs))