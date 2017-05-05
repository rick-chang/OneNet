
import numpy as np
import scipy as sp
from vec import vec
import matplotlib.pyplot as plt


""" currently only support width (and height) * resize_ratio is an interger! """
def setup(x_shape, box_size, total_box = 1):

    spare = 0.25 * box_size

    mask = np.ones(x_shape)

    for i in range(total_box):

        start_row = spare
        end_row = x_shape[1] - spare - box_size - 1
        start_col = spare
        end_col = x_shape[2] - spare - box_size - 1

        idx_row = int(np.random.rand(1) * (end_row - start_row) + start_row)
        idx_col = int(np.random.rand(1) * (end_col - start_col) + start_col)

        mask[0,idx_row:idx_row+box_size,idx_col:idx_col+box_size,:] = 0.


    def A_fun(x):
        y = np.multiply(x, mask);
        return y

    def AT_fun(y):
        x = np.multiply(y, mask);
        return x

    return (A_fun, AT_fun, mask)


