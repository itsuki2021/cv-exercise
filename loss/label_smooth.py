import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(preds: Tensor, target: Tensor, reduction):
    """ Cross Entropy Loss, E{x~P}[-log(Q(x))]

    :param preds:       tensor(B, N)
    :param target:      tensor(B, N)
    :param reduction:   reduction manner for batch calculation
    :return:            cross entropy
    """
    log_q = F.log_softmax(preds, dim=1)
    loss = torch.sum(-log_q * target, dim=1)
    assert reduction in ('none', 'mean', 'sum'), '`reduction` must be one of \'none\', \'mean\', or \'sum\'.'

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


def one_hot_encoding(labels: Tensor, n_classes: int):
    """

    :param labels:      tensor(size=(B, 1)), e.g. tensor([1, 3, 3, 5])
    :param n_classes:   number of classes
    :return:            one-hot encoding, e.g.
                        tensor([[0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1]])
    """
    return torch.zeros(labels.size(0), n_classes).to(labels.device).scatter_(
        dim=1, index=labels.view(-1, 1), value=1)


def label_smoothing(preds: Tensor, targets: Tensor, epsilon=0.1):
    n_classes = preds.size(1)
    device = preds.device

    one_hot = one_hot_encoding(targets, n_classes).float().to(device)
    targets = one_hot * (1 - epsilon) + torch.ones_like(one_hot).to(device) * epsilon / n_classes
    loss = cross_entropy_loss(preds, targets, reduction="mean")

    return loss
