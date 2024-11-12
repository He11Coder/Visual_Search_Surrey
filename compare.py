import numpy as np

def L1Compare(F1, F2):
    abs_diffs = np.abs(np.subtract(F1, F2))

    return np.sum(abs_diffs)


def L2Compare(F1, F2):
    square_diffs = np.square(np.subtract(F1, F2))

    return np.sqrt(np.sum(square_diffs))


def LInftyCompare(F1, F2):
    abs_diffs = np.abs(np.subtract(F1, F2))

    return np.max(abs_diffs)


def CosineCompare(F1, F2):
    dot_prod = np.dot(F1, F2)
    
    return 1.0 - np.divide(dot_prod, np.multiply(np.linalg.norm(F1), np.linalg.norm(F2)))