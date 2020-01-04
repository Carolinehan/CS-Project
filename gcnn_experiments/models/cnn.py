import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import ricnn

VALID='VALID'

filter_size = 3
n_filters = 128
stride = 1
pool_stride = 2

def build_layer(x_depth, layer_order, previous_layer, weight_size=n_filters,  activation='relu'):
    weights = tf.get_variable('weights-conv'+str(layer_order), [filter_size,filter_size, x_depth, weight_size])
    biases = tf.get_variable('biases-conv'+str(layer_order), [weight_size])
    output = tf.nn.conv2d(previous_layer, weights, padding=VALID, use_cudnn_on_gpu=True, data_format='NHWC')
    output = tf.nn.bias_add(output, biases)
    output = tf.layers.batch_normalization(output)
    if activation == 'relu':
        output = tf.nn.relu(output)
    elif activation == 'softmax':
        output = tf.nn.softmax(output)
    return output

def get_model(x,x_depth, y_size):
    l1 = build_layer(x_depth,1,x)
    l2 = build_layer(x_depth,2,l1)
    l2 = tf.nn.max_pool(l2, [1, pool_stride, pool_stride, 1],
                                [1, pool_stride, pool_stride, 1], padding='VALID')

    l3 = build_layer(x_depth,3,l2)
    l4 = build_layer(x_depth,4,l3)
    l5 = build_layer(x_depth,5,l4)
    l6 = build_layer(x_depth,6,l5)
    images_flat = tf.contrib.layers.flatten(l6)
    l7 = tf.contrib.layers.fully_connected(images_flat, y_size, tf.nn.softmax)
    return l7


