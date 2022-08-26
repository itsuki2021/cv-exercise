from typing import Optional
import numpy as np


class Linear:
    def __init__(self, dim_in: int, dim_out: int):
        scale = np.sqrt(dim_in / 2)
        self.weight = np.random.standard_normal(size=(dim_out, dim_in)) / scale
        self.bias = np.random.standard_normal(size=dim_out) / scale
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray):
        """ Forward computation

        :param x:   input data, shape: (B, dim_in)
        :return:    output data, shape: (B, dim_out)
        """
        return x @ self.weight.T + self.bias

    def __call__(self, x: np.ndarray):
        self._x = x
        return self.forward(x)

    def backward(self, d_out: np.ndarray):
        """ Forward computation

        :param d_out:   output gradient(dL / dy), shape: (B, dim_out)
        :return:        gradient of weight and bias
        """
        if self._x is None:
            raise TypeError("Can not compute gradient before calling 'forward()'")

        d_x = d_out @ self.weight       # output gradient of previous layer
        d_w = d_out.T @ self._x
        d_b = np.mean(d_out, axis=0)

        return d_x, d_w, d_b
