import numpy as np


def compute_iou(bb1: np.ndarray, bb2: np.ndarray):
    """ Compute Batch IoU

    :param bb1:     array(N, 4+), [[x_min, y_min, x_max, y_max, ...], ...]
    :param bb2:     array(N, 4+)
    :return:        IoUs
    """
    left = np.maximum(bb1[:, 0], bb2[:, 0])
    right = np.minimum(bb1[:, 2], bb2[:, 2])
    top = np.maximum(bb1[:, 1], bb2[:, 1])
    bottom = np.minimum(bb1[:, 3], bb2[:, 3])
    s1 = (bb1[:, 2] - bb1[:, 0]) * (bb1[:, 3], bb1[:, 1])
    s2 = (bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3], bb2[:, 1])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    ans = intersect.astype(float) / (s1 + s2 - intersect).astype(float)

    return ans