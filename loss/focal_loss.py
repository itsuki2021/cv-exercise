import torch
from torch import Tensor
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fun = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = self.loss_fun.reduction
        self.loss_fun.reduction = 'none'

    def forward(self, pred: Tensor, true: Tensor):
        """

        :param pred:    tensor(N, num_classes)
        :param true:    tensor(N, num_classes)
        :return:        focal loss
        """
        loss = self.loss_fun(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:   # 'none'
            return loss
