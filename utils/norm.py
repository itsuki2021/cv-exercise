import numpy as np


def batch_norm_forward(x: np.ndarray,
                       gamma: float,
                       beta: float,
                       mode='train',
                       eps=1e-8,
                       momentum=0.9,
                       running_mean=None,
                       running_var=None):
    """ Batch normalization in forward

    :param x:               input array with shape (B, N)
    :param gamma:           gamma in BN
    :param beta:            beta in BN
    :param mode:            'train' or 'test'
    :param eps:             epsilon in BN
    :param momentum:        used in mean variance updating
    :param running_mean:    mean value of training set
    :param running_var:     variance of training set
    :return:                batch normalize result
    """
    B, D = x.shape
    if running_mean is None:
        running_mean = np.zeros(D, dtype=x.dtype)
    if running_var is None:
        running_var = np.zeros(D, dtype=x.dtype)

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        cache = (x, gamma, beta, x_hat, sample_mean, sample_var, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = running_var * momentum + (1 - momentum) * sample_var
    elif mode == 'test':
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
    else:
        raise ValueError('Invalid forward batch-norm mode "%s"' % mode)

    return out, cache, running_mean, running_var


def batch_norm_backward(d_out, cache):
    """ Batch normalization in backward

    :param d_out:   gradient of next layer, dL / dy, BxD
    :param cache:   cache from BN-forward
    :return:        gradient of x, gamma and beta
    """
    x, gamma, beta, x_hat, sample_mean, sample_var, eps = cache
    B = x.shape[0]

    d_x_hat = d_out * gamma
    d_sigma = -0.5 * np.sum(d_x_hat * (x - sample_mean), axis=0) * np.power(sample_var + eps, -1.5)
    d_mu = -np.sum(d_x_hat / np.sqrt(sample_var + eps), axis=0) - 2 * d_sigma * np.sum(x - sample_mean, axis=0) / B
    d_x = d_x_hat / np.sqrt(sample_var + eps) + 2.0 * d_sigma * (x - sample_mean) / B + d_mu / B
    d_gamma = np.sum(d_out * x_hat, axis=0)
    d_beta = np.sum(d_out, axis=0)

    return d_x, d_gamma, d_beta
