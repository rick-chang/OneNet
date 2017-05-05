
import numpy as np
import scipy as sp
from vec import vec
import matplotlib.pyplot as plt


def setup(x_shape, box_size):


    mask = np.ones(x_shape)


    idx_row = np.round(float(x_shape[1]) / 2.0 - float(box_size) / 2.0).astype(int)
    idx_col = np.round(float(x_shape[2]) / 2.0 - float(box_size) / 2.0).astype(int)

    mask[0,idx_row:idx_row+box_size,idx_col:idx_col+box_size,:] = 0.


    def A_fun(x):
        y = np.multiply(x, mask);
        return y

    def AT_fun(y):
        x = np.multiply(y, mask);
        return x

    return (A_fun, AT_fun, mask)


