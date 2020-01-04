import os
import numpy as np
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
import time
import matplotlib.pyplot as plt

VALID='VALID'
channels = 10
best_model='best_model'
results_fold='results'


def generate_p4z2layer(layer_order, previous_layer, x_depth, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='Z2', h_output='C4', in_channels=x_depth, out_channels=channels, ksize=3)
    w = tf.get_variable('weights-conv' + str(layer_order), w_shape)
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l = tf.layers.batch_normalization(l)
    l = tf.nn.relu(l)
    return l

def generate_p4p4layer(layer_order, previous_layer, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='C4', h_output='C4', in_channels=channels, out_channels=channels, ksize=kernel_size)
    w = tf.get_variable('weights-conv' + str(layer_order), w_shape)
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    return l

def generate_p4p4layer_n_a(layer_order, previous_layer, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='C4', h_output='C4', in_channels=channels, out_channels=channels, ksize=kernel_size)
    w = tf.get_variable('weights-conv' + str(layer_order), w_shape)
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l = tf.layers.batch_normalization(l, training=True)
    l = tf.nn.relu(l)
    return l

def get_model(x, x_depth, y_size):
    l1=generate_p4z2layer(1, x, x_depth)
    l2 =generate_p4p4layer_n_a(2, l1)
    l2= tf.nn.max_pool(l2,[1,2, 2,1], [1,2, 2,1], padding=VALID)
    l3 = generate_p4p4layer_n_a(3,l2)
    l4 = generate_p4p4layer_n_a(4, l3)
    l5 = generate_p4p4layer_n_a(5, l4)
    l6 = generate_p4p4layer_n_a(6, l5)
    l7 = generate_p4p4layer_n_a(7, l6, kernel_size=4 )
    images_flat = tf.contrib.layers.flatten(l7)
    l7 = tf.contrib.layers.fully_connected(images_flat, y_size, tf.nn.softmax)
    return l7