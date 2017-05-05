
import numpy as np
import scipy as sp
from vec import vec
import matplotlib.pyplot as plt


""" currently only support width (and height) * resize_ratio is an interger! """
def setup(x_shape, resize_ratio):

    box_size = 1.0 / resize_ratio
    if np.mod(x_shape[1], box_size) != 0 or np.mod(x_shape[2], box_size) != 0:
        print "only support width (and height) * resize_ratio is an interger!"


    def A_fun(x):
        y = box_average(x, int(box_size))
        return y

    def AT_fun(y):
        x = box_repeat(y, int(box_size))
        return x

    return (A_fun, AT_fun)



def box_average(x, box_size):
    """ x: [1, row, col, channel] """
    im_row = x.shape[1]
    im_col = x.shape[2]
    channel = x.shape[3]
    out_row = np.floor(float(im_row) / float(box_size)).astype(int)
    out_col = np.floor(float(im_col) / float(box_size)).astype(int)
    y = np.zeros((1,out_row,out_col,channel))
    total_i = int(im_row / box_size)
    total_j = int(im_col / box_size)

    for c in range(channel):
        for i in range(total_i):
            for j in range(total_j):
                avg = np.average(x[0, i*int(box_size):(i+1)*int(box_size), j*int(box_size):(j+1)*int(box_size), c], axis=None)
                y[0,i,j,c] = avg

    return y


def box_repeat(x, box_size):
    """ x: [1, row, col, channel] """
    im_row = x.shape[1]
    im_col = x.shape[2]
    channel = x.shape[3]
    out_row = np.floor(float(im_row) * float(box_size)).astype(int)
    out_col = np.floor(float(im_col) * float(box_size)).astype(int)
    y = np.zeros((1,out_row,out_col,channel))
    total_i = im_row
    total_j = im_col

    for c in range(channel):
        for i in range(total_i):
            for j in range(total_j):
                y[0, i*int(box_size):(i+1)*int(box_size), j*int(box_size):(j+1)*int(box_size), c] = x[0,i,j,c]
    return y