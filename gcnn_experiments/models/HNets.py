'''MNIST-rot model'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf

import harmonic_network_lite as hn_lite
#from harmonic_network_helpers import *

def get_model(x, x_size, x_depth, y_size, train_phase):
   """The MNIST-rot model similar to the one in Cohen & Welling, 2016"""
   # Sure layers weight & bias
   order = 1
   # Number of Filters
   nf =8
   nf2 = int(nf*2)
   nf3 = int(nf*(2**2.))
   # bs = 46
   bs = 100
   fs = 5
   ncl = y_size
   sm = 0.7
   nr = 4

   # Create bias for final layer
   bias = tf.compat.v1.get_variable('b7', shape=[y_size],
                     initializer=tf.constant_initializer(1e-2))
   x = tf.reshape(x, shape=[bs,x_size,x_size,1,1,x_depth])

   # Convolutional Layers with pooling
   with tf.name_scope('block1') as scope:
      cv1 = hn_lite.conv2d(x, nf, fs, padding='SAME', n_rings=nr, name='1')
      cv1 = hn_lite.non_linearity(cv1, tf.nn.relu, name='1')

      cv2 = hn_lite.conv2d(cv1, nf, fs, padding='SAME', n_rings=nr, name='2')
      cv2 = hn_lite.batch_norm(cv2, train_phase, name='bn1')

   with tf.name_scope('block2') as scope:
      cv2 = hn_lite.mean_pool(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
      cv3 = hn_lite.conv2d(cv2, nf2, fs, padding='SAME', n_rings=nr, name='3')
      cv3 = hn_lite.non_linearity(cv3, tf.nn.relu, name='3')

      cv4 = hn_lite.conv2d(cv3, nf2, fs, padding='SAME', n_rings=nr, name='4')
      cv4 = hn_lite.batch_norm(cv4, train_phase, name='bn2')

   with tf.name_scope('block3') as scope:
      cv4 = hn_lite.mean_pool(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
      cv5 = hn_lite.conv2d(cv4, nf3, fs, padding='SAME', n_rings=nr, name='5')
      cv5 = hn_lite.non_linearity(cv5, tf.nn.relu, name='5')

      cv6 = hn_lite.conv2d(cv5, nf3, fs, padding='SAME', n_rings=nr, name='6')
      cv6 = hn_lite.batch_norm(cv6, train_phase, name='bn3')

   # Final Layer
   with tf.name_scope('block4') as scope:
      cv7 = hn_lite.conv2d(cv6, ncl, fs, padding='SAME', n_rings=nr, phase=False,
               name='7')
      real = hn_lite.sum_magnitudes(cv7)
      cv7 = tf.reduce_mean(real, axis=[1,2,3,4])
      return tf.nn.bias_add(cv7, bias)
