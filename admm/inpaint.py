
import numpy as np
from vec import vec


def setup(x_shape, drop_prob = 0.5):

    mask = np.random.rand(*x_shape) > drop_prob;
    mask = mask.astype('double')

    def A_fun(x):
        y = np.multiply(x, mask);
        return y

    def AT_fun(y):
        x = np.multiply(y, mask);
        return x

    return (A_fun, AT_fun, mask)