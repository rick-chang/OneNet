import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as tf_batch_norm
import tensorflow.contrib.slim as slim

def new_fc_layer(bottom, output_size, name=None, bias=True):
    """
    fully connected layer
    """
    shape = bottom.get_shape().as_list()
    dim = np.prod( shape[1:] )
    x = tf.reshape( bottom, [-1, dim])
    input_size = dim

    with tf.variable_scope(name):
        w = tf.get_variable(
                "W",
                shape=[input_size, output_size],
                initializer=tf.truncated_normal_initializer(0., 0.005))
        if bias == True:
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)
        else:
            fc = tf.matmul(x, w)

    return (fc, w)

def batchnorm(bottom, is_train, num_reference, epsilon=1e-3, decay=0.999, name=None):
    """ virtual batch normalization (poor man's version)
    the first half is the true batch, the second half is the reference batch.
    When num_reference = 0, it is just typical batch normalization.  
    To use virtual batch normalization in test phase, "update_popmean.py" needed to be executed first 
    (in order to store the mean and variance of the reference batch into pop_mean and pop_variance of batchnorm.)
    """

    batch_size = bottom.get_shape().as_list()[0]
    inst_size = batch_size - num_reference
    instance_weight = np.ones([batch_size])

    if inst_size > 0:
        reference_weight = 1.0 - (1.0 / ( num_reference + 1.0))
        instance_weight[0:inst_size] = 1.0 - reference_weight
        instance_weight[inst_size:] = reference_weight
    else:
        decay = 0.0

    return slim.batch_norm(bottom, activation_fn=None, is_training=is_train, decay=decay, scale=True, scope=name, batch_weights=instance_weight)


def new_conv_layer(bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, bias=True, name=None):
    """
    typical convolution layer using stride to down-sample
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            "W",
            shape=filter_shape,
            initializer=tf.truncated_normal_initializer(0., 0.005))
        conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)

        if bias == True:
            b = tf.get_variable(
                "b",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.))
            output = activation(tf.nn.bias_add(conv, b))
        else:
            output = activation(conv)

    return output


def new_deconv_layer(bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, bias=True, name=None):
    """
    typical deconvolution layer
    """    
    with tf.variable_scope(name):
        W = tf.get_variable(
            "W",
            shape=filter_shape,
            initializer=tf.truncated_normal_initializer(0., 0.005))
        deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)

        if bias == True:
            b = tf.get_variable(
                "b",
                shape=filter_shape[-2],
                initializer=tf.constant_initializer(0.))
            output = activation(tf.nn.bias_add(deconv, b))
        else:
            output = activation(deconv)

    return output


def channel_wise_fc_layer(bottom, name, bias=True):
    """
    channel-wise fully connected layer
    """
    _, width, height, n_feat_map = bottom.get_shape().as_list()
    input_reshape = tf.reshape( bottom, [-1, width*height, n_feat_map] )  # order='C'
    input_transpose = tf.transpose( input_reshape, [2,0,1] )  # n_feat_map * batch * d

    with tf.variable_scope(name):
        W = tf.get_variable(
                "W",
                shape=[n_feat_map,width*height, width*height], # n_feat_map * d * d_filter
                initializer=tf.truncated_normal_initializer(0., 0.005))
        output = tf.batch_matmul(input_transpose, W)  # n_feat_map * batch * d_filter

        if bias == True:
            b = tf.get_variable(
                "b",
                shape=width*height,
                initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)

    output_transpose = tf.transpose(output, [1,2,0])  # batch * d_filter * n_feat_map
    output_reshape = tf.reshape( output_transpose, [-1, width, height, n_feat_map] )
    return output_reshape



def bottleneck(input, is_train, n_reference, channel_compress_ratio=4, stride=1, bias=True, name=None):
    """
    building block for creating residual net
    """
    input_shape = input.get_shape().as_list()

    if stride is not 1:
        output_channel = input_shape[3] * 2
    else:
        output_channel = input_shape[3]

    bottleneck_channel = output_channel / channel_compress_ratio

    with tf.variable_scope(name):
        bn1 = tf.nn.elu(batchnorm(input, is_train, n_reference, name='bn1'))
        # shortcut
        if stride is not 1:
            shortcut = new_conv_layer(bn1, [1,1,input_shape[3],output_channel], stride=stride, bias=bias, name="conv_sc" )
        else:
            shortcut = input

        # bottleneck_channel
        conv1 = new_conv_layer(bn1, [1,1,input_shape[3],bottleneck_channel], stride=stride, bias=bias, name="conv1" )
        bn2 = tf.nn.elu(batchnorm(conv1, is_train, n_reference, name='bn2'))
        conv2 = new_conv_layer(bn2, [3,3,bottleneck_channel,bottleneck_channel], stride=1, bias=bias, name="conv2" )
        bn3 = tf.nn.elu(batchnorm(conv2, is_train, n_reference, name='bn3'))
        conv3 = new_conv_layer(bn3, [1,1,bottleneck_channel,output_channel], stride=1, bias=bias, name="conv3" )

    return shortcut+conv3



def bottleneck_flexible(input, is_train, output_channel, n_reference, channel_compress_ratio=4, stride=1, bias=True, name=None):

    input_shape = input.get_shape().as_list()

    bottleneck_channel = output_channel / channel_compress_ratio

    with tf.variable_scope(name):
        bn1 = tf.nn.elu(batchnorm(input, is_train, n_reference, name='bn1'))
        # shortcut
        if stride is not 1:
            shortcut = new_conv_layer(bn1, [1,1,input_shape[3],output_channel], stride=stride, bias=bias, name="conv_sc" )
        else:
            shortcut = input

        # bottleneck_channel
        conv1 = new_conv_layer(bn1, [1,1,input_shape[3],bottleneck_channel], stride=stride, bias=bias, name="conv1" )
        bn2 = tf.nn.elu(batchnorm(conv1, is_train, n_reference, name='bn2'))
        conv2 = new_conv_layer(bn2, [3,3,bottleneck_channel,bottleneck_channel], stride=1, bias=bias, name="conv2" )
        bn3 = tf.nn.elu(batchnorm(conv2, is_train, n_reference, name='bn3'))
        conv3 = new_conv_layer(bn3, [1,1,bottleneck_channel,output_channel], stride=1, bias=bias, name="conv3" )

    return shortcut+conv3



def add_bottleneck_module(input, is_train, nBlocks, n_reference, channel_compress_ratio=4, bias=True, name=None):

    with tf.variable_scope(name):
        # the first block reduce spatial dimension
        out = bottleneck(input, is_train, n_reference, channel_compress_ratio=channel_compress_ratio, stride=2, bias=bias, name='block0')

        for i in range(nBlocks-1):
            subname = 'block%d' % (i+1)
            out = bottleneck(out, is_train, n_reference, channel_compress_ratio=channel_compress_ratio, stride=1, bias=bias, name=subname)
    return out

