import numpy as np


def pca(x: np.ndarray, k: int):
    """ Primary Component Analysis

    :param x:   input array, N d-dimensional vector
    :param k:   dimension(s) after pca transform
    :return:    pca result
    """
    d, N = x.shape
    assert 1 <= k <= d
    x -= np.mean(x, axis=1)[:, None].repeat(N, axis=1)
    co_var = x @ x.T
    eig_val, eig_vec = np.linalg.eig(co_var)
    ind = np.argsort(eig_val)[::-1]
    trans_mat = eig_vec[:, :k]  # dxk

    return trans_mat.T @ x
