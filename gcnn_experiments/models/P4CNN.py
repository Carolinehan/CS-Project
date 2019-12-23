import os
import numpy as np
import tensorflow as tf
import time
import logging
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util

VALID='VALID'
channels = 10

def generate_p4z2layer(previous_layer, iamge_channel, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='Z2', h_output='D4', in_channels=iamge_channel, out_channels=channels, ksize=3)
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l = tf.layers.batch_normalization(l)
    l = tf.nn.relu(l)
    return l

def generate_p4p4layer(previous_layer, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='D4', h_output='D4', in_channels=channels, out_channels=channels, ksize=kernel_size)
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    return l

def generate_p4p4layer_n_a(previous_layer, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='D4', h_output='D4', in_channels=channels, out_channels=channels, ksize=kernel_size)
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l = tf.layers.batch_normalization(l, training=True)
    l = tf.nn.relu(l)
    return l

def get_model(x, iamge_channel):
    l1=generate_p4z2layer(x, iamge_channel)
    l2 =generate_p4p4layer_n_a(l1)
    l2= tf.nn.max_pool(l2,[1,2, 2,1], [1,2, 2,1], padding=VALID)
    l3 = generate_p4p4layer_n_a(l2)
    l4 = generate_p4p4layer_n_a(l3)
    l5 = generate_p4p4layer_n_a(l4)
    l6 = generate_p4p4layer_n_a(l5)
    l7=generate_p4p4layer(l6)
    return l6

datadir ='../mnist-rot'
trainfn='train.npz'
valfn='valid.npz'
testfn='test.npz'
def preprocess_mnist_data(train_data, test_data, train_labels, test_labels):
    train_mean = np.mean(train_data)  # compute mean over all pixels make sure equivariance is preserved
    train_data -= train_mean
    test_data -= train_mean
    train_std = np.std(train_data)
    train_data /= train_std
    test_data /= train_std
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    return train_data, test_data, train_labels, test_labels
def create_result_dir(modelfn, logme):
    # if args.restart_from is None:
    result_dir = os.path.join(datadir, os.path.basename(modelfn).split('.')[0])
    result_dir += '/' + time.strftime('r%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)
    logging.info(logme)

    # Create init file so we can import the model module
    f = open(os.path.join(result_dir, '__init__.py'), 'wb')
    f.close()

    return log_fn, result_dir



def read_data(file_name):
    set = np.load(os.path.join(datadir, file_name))
    data = set['data']
    data = np.reshape(data, (-1, 28, 28, 1))
    labels = set['labels']
    return data, labels

def get_data():
    train_data, train_labels = read_data(trainfn)
    val_data, val_labels = read_data(valfn)
    train_data, val_data, train_labels, val_labels = preprocess_mnist_data(
        train_data, val_data, train_labels, val_labels)

    test_data, test_labels = read_data(testfn)
    _, test_data, _, test_labels = preprocess_mnist_data(
        train_data, test_data, train_labels, test_labels)

    return train_data,train_labels, val_data, val_labels, test_data, test_labels

x_size = 28
x_depth = 1
batch_size = 100
y_size = 10
epochs=100

def get_acc(data, labels, x, y, accuracy, type):
    val_step=0
    val_acc=0
    for i in range(0, len(data), batch_size):
        # Get the next batch
        input_batch = data[i:i + batch_size, :, :, :]
        labels_batch = labels[i:i + batch_size]
        feed_dict = {x: input_batch, y: labels_batch}

        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        val_step += 1
        val_acc += train_accuracy
    acc = val_acc / val_step
    print("%s accuracy %g" % (type, acc * 100))

def train():
    train_data, train_labels, val_data, val_labels, test_data, test_labels = get_data()
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, x_size, x_size, x_depth])
        y = tf.placeholder(dtype=tf.int64, shape=[None])
        output = get_model(x, x_depth)
        images_flat = tf.contrib.layers.flatten(output)
        logits = tf.contrib.layers.fully_connected(images_flat, y_size, tf.nn.relu)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        correct_prediction = tf.equal(y, tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print('start training epoch %g' % (epoch))
            for i in range(0, len(train_data), batch_size):
                # Get the next batch
                input_batch = train_data[i:i+batch_size, :,:,:]
                labels_batch = train_labels[i:i+batch_size]
                feed_dict = {x: input_batch, y: labels_batch}

                # Run the training step
                train_op.run(feed_dict=feed_dict)
                loss_percent=loss.eval(feed_dict=feed_dict)
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                print("Step %d, training batch accuracy %g  loss %f" % (i,  train_accuracy * 100, loss_percent))

            get_acc(val_data, val_labels, x, y, accuracy, 'validation')
    get_acc(test_data, test_labels, x, y, accuracy, 'test')

train()
