import numpy as np
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
import os
import numpy as np
import tensorflow as tf
import time
import logging
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling3D,GlobalMaxPooling3D, MaxPooling1D, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import add
from keras.activations import relu, softmax,sigmoid
from keras.models import Model
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,EarlyStopping

VALID='VALID'
channels = 10

def generate_layer(previous_layer, kernel_size=3):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='D4', h_output='D4', in_channels=channels, out_channels=channels, ksize=kernel_size)
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
    l = gconv2d(input=previous_layer, filter=w, strides=[1, 1, 1, 1], padding=VALID,
                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l = tf.layers.batch_normalization(l, training=True)
    l = tf.nn.relu(l)
    return l

def get_model(x, image_chanell):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input='Z2', h_output='D4', in_channels=image_chanell, out_channels=channels, ksize=3)
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
    l1 = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding=VALID,
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info, use_cudnn_on_gpu=True)
    l1 = tf.layers.batch_normalization(l1)
    l1 =  tf.nn.relu(l1)
    l2 =generate_layer(l1)
    #l2= tf.layers.MaxPooling2D((2, 2), (2, 2), padding=VALID)(l2)
    l3 = generate_layer(l2)
    l4 = generate_layer(l3)
    l5 = generate_layer(l4)
    l6 = generate_layer(l5)
    return l6

datadir ='../mnist-rot'
trainfn='train.npz'
valfn='valid.npz'
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

x_size = 28
x_depth = 1
batch_size = 100
y_size = 10
def train():
    train_set = np.load(os.path.join(datadir, trainfn))
    val_set = np.load(os.path.join(datadir, valfn))
    train_data = train_set['data']
    train_data =  np.reshape(train_data, (-1,28,28,1))
    train_labels = train_set['labels']
    val_data = val_set['data']
    val_data = np.reshape(val_data, (-1, 28, 28, 1))
    val_labels = val_set['labels']
    train_data, val_data, train_labels, val_labels = preprocess_mnist_data(
        train_data, val_data, train_labels, val_labels)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, x_size, x_size, x_depth])
        y = tf.placeholder(dtype=tf.int64, shape=[None])

        output = get_model(x, 1)
        images_flat = tf.contrib.layers.flatten(output)
        logits = tf.contrib.layers.fully_connected(images_flat, y_size, tf.nn.relu)
        predicted_labels = tf.argmax(logits, 1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        init = tf.global_variables_initializer()
        correct_prediction = tf.equal(y, tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Run the training
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        learning_rate = 0.05
        steps_number = 100
        step_sizes=100
        for i in range(steps_number):
            # Get the next batch
            input_batch = train_data[i*step_sizes:(i+1)*step_sizes, :,:,:]
            labels_batch = train_labels[i*step_sizes:(i+1)*step_sizes]
            feed_dict = {x: input_batch, y: labels_batch}

            # Run the training step
            train_op.run(feed_dict=feed_dict)

            # Print the accuracy progress on the batch every 100 steps
            if i % 10 == 0:
                # if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                print("Step %d, training batch accuracy %g %%" % (i, train_accuracy * 100))

        # Evaluate on the test set
        # test_data = mnist.test.images
        # test_data = np.reshape(test_data[:100, :], (-1, 28, 28, 1))
        # test_label = mnist.test.labels[:100, :]
        # test_accuracy = accuracy.eval(feed_dict={x: test_data, y: test_label})
        # print("Test accuracy: %g %%" % (test_accuracy * 100))
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.0/lib64"
a=tf.test.is_gpu_available()
train()
