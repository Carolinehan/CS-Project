import os
import numpy as np
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
import time
import matplotlib.pyplot as plt
import harmonic_network_ops as ho

VALID='VALID'
channels = 10
best_model='best_model'
results_fold='results'
batch_size = 100


def generate_p4z2layer(layer_order, previous_layer, x_depth, train_phase, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='Z2', h_output='C4', in_channels=x_depth, out_channels=channels, ksize=3)
    w = tf.get_variable('weights-conv' + str(layer_order), w_shape)
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l = ho.bn(l, train_phase=train_phase, name='bn'+str(layer_order))
    l = tf.nn.relu(l)
    return l

def generate_p4p4layer(layer_order, previous_layer, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='C4', h_output='C4', in_channels=channels, out_channels=channels, ksize=kernel_size)
    w = tf.get_variable('weights-conv' + str(layer_order), w_shape)
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    return l

def generate_p4p4layer_n_a(layer_order, previous_layer, train_phase, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='C4', h_output='C4', in_channels=channels, out_channels=channels, ksize=kernel_size)
    w = tf.get_variable('weights-conv' + str(layer_order), w_shape)

    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l = ho.bn(l, train_phase=train_phase, name='bn'+str(layer_order))
    l = tf.nn.relu(l)
    return l

def get_model(x, x_depth, y_size, train_phase):
    l1=generate_p4z2layer(1, x, x_depth, train_phase)
    l2 =generate_p4p4layer_n_a(2, l1,train_phase)
    l2= tf.nn.max_pool(l2,[1,2, 2,1], [1,2, 2,1], padding=VALID)
    l3 = generate_p4p4layer_n_a(3,l2,train_phase)
    l4 = generate_p4p4layer_n_a(4, l3,train_phase)
    l5 = generate_p4p4layer_n_a(5, l4,train_phase)
    l6 = generate_p4p4layer_n_a(6, l5,train_phase)
    l7 = generate_p4p4layer_n_a(7, l6,train_phase, kernel_size=4)
    images_flat = tf.contrib.layers.flatten(l7)
    w_shape = (int(images_flat.shape[-1]), y_size)
    weights = tf.get_variable('weights-full', w_shape)
    biases = tf.get_variable('biases-full', [y_size])
    l7 = tf.matmul(images_flat, weights)+biases
    return l7