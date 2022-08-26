import numpy as np


def hist_eq(img_gray: np.ndarray) -> np.ndarray:
    assert len(img_gray.shape) == 2, "Can not deal with multi-channel image!"
    h, w = img_gray.shape
    total_num = h * w

    pr = np.zeros(shape=(256,))
    ps = np.zeros(shape=(256,))

    for i in range(256):
        pr[i] = np.sum(img_gray == i) / total_num

    sum = 0
    for i in range(256):
        sum += pr[i]
        ps[i] = sum

    img_eq = img_gray.copy()
    for i in range(256):
        img_eq[img_gray == i] = (ps[i] * 255)

    return img_eq