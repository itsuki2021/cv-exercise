import numpy as np


class SGD:
    def __init__(self, params: np.ndarray, lr: float, momentum: float = None):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = []

        if momentum is not None:
            for param in self.params:
                self.velocity.append(np.zeros_like(param))

    def update_params(self, grads):
        if self.momentum is None:
            for param, grad in zip(self.params, grads):
                param -= self.lr * grad
        else:
            for i in range(len(self.params)):
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
                self.params[i] += self.velocity[i]
