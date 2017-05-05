
import numpy as np
import scipy as sp
from vec import vec
import matplotlib.pyplot as plt



def setup(x_shape, compress_ratio):

    d = np.prod(x_shape).astype(int)
    m = np.round(compress_ratio * d).astype(int)

    A = np.random.randn(m,d) / np.sqrt(m)


    def A_fun(x):
        y = np.dot(A, x.ravel(order='F'))
        y = np.reshape(y, [1, m], order='F')
        return y

    def AT_fun(y):
        y = np.reshape(y, [m, 1], order='F')
        x = np.dot(A.T, y)
        x = np.reshape(x, x_shape, order='F')
        return x

    return (A_fun, AT_fun, A)

