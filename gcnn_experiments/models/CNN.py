import tensorflow as tf
import numpy as np
import statics

x=tf.placeholder(tf.float32, [None, 28,28])
y=tf.placeholder(tf.float32, [None, len(names)])


def conv_layer(x,w,b):
    conv=tf.nn.conv2d(x,w,strides=[1,1,1,1], padding=statics.PADDING_VALID)
    conv_with_b = tf.nn.bias_add(conv,b)
    conv_out=tf.nn.relu(conv_with_b)
    return conv_out

def maxpool_layer(conv, k=2):
    return tf.nn.max_pool(conv, ksize=[1,k,k,1], strides=[1,k,k,1], padding=statics.PADDING_VALID)

def model():
    w1=tf.Variable(tf.random_normal([5,5,1,64]))
    b1=tf.Variable(tf.random_normal([64]))
    l1=conv_layer(x,w1,b1)
    l1=maxpool_layer(l1)
    l1=tf.nn.lrn(l1,4,bias=1.0, alpha=0.001/9.0, beta=0.75)

    w2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
    b2 = tf.Variable(tf.random_normal([64]))

    l2 = conv_layer(l1, w2, b2)
    l2 = tf.nn.lrn(l2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    l2 = maxpool_layer(l2)

    w3 = tf.Variable(tf.random_normal([6*6*64, 1024]))
    b3 = tf.Variable(tf.random_normal([1024]))
    l3= tf.reshape(l2, [-1, w3.get_shape().as_list()[0]])
    l3=tf.add(tf.matmul(l3,w3),b3)
    l3=tf.nn.relu(l3)

    w_out = tf.Variable(tf.random_normal([1024, len(names)]))
    b_out = tf.Variable(tf.random_normal([len(names)]))

    out = tf.add(tf.matmul(l3,w_out), b_out)

    return out