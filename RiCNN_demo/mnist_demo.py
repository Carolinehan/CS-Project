import tensorflow as tf
import numpy as np
import ricnn
import h5py
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_size = 28
x_depth = 1
batch_size = 100
y_size = 10

WORK_DICTIONARY ='dataset/mnist/'
TRAIN_DATA='train.hdf5'
VALIDATION_DATA='validation.hdf5'
TEST_DATA='test.hdf5'

def get_data(file_name):
    f = h5py.File(os.path.join(WORK_DICTIONARY, file_name), 'r')
    data = f['data']
    labels =np.array(f['labels']).astype('int')
    one_hot = np.eye(y_size)[labels]
    return data, one_hot


with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, x_size, x_size, x_depth])
    y = tf.placeholder(dtype=tf.int32, shape=[None,y_size])
    # Rotation-equivariant convolution w/ stride
    filter_size = 3
    n_filters = 16
    n_rotations = 4
    stride = 2
    pool_stride = 1
    weights = tf.get_variable('weights-conv1', [filter_size, filter_size,
                                                x_depth, n_filters])
    biases = tf.get_variable('biases-conv1', [n_filters])
    output = ricnn.rotation_equivariant_conv2d(
            x, weights,

            [batch_size, x_size, x_size, x_depth],
            [filter_size, filter_size, x_depth, n_filters],
            n_rotations, stride=stride) + biases
    output = tf.nn.relu(output)
    output_size = ricnn.calculate_reconv_output_size(
            x_size, filter_size, stride=stride)
    n_filters_previous_layer = n_filters

    # Rotation-equivariant convolution w/ max pooling
    filter_size = 3
    n_filters = 16
    n_rotations = 4
    stride = 1
    pool_stride = 2
    weights = tf.get_variable('weights-conv2', [filter_size, filter_size,
                                                n_filters_previous_layer, n_filters])
    biases = tf.get_variable('biases-conv2', [n_filters])
    output = ricnn.rotation_equivariant_conv2d(
            output, weights,
            [batch_size, output_size, output_size, n_filters_previous_layer],
            [filter_size, filter_size, n_filters_previous_layer, n_filters],
            n_rotations, stride=stride) + biases
    output = tf.nn.max_pool(output, [1, pool_stride, pool_stride, 1],
                            [1, pool_stride, pool_stride, 1], padding='VALID')
    output = tf.nn.relu(output)
    output_size = ricnn.calculate_reconv_output_size(
            output_size, filter_size, pool_stride=pool_stride)
    n_filters_previous_layer = n_filters

    # Rotation-invariant Conv-to-Full transition with the 2D-DFT
    n_filters = 16
    n_rotations = 4
    filter_size = output_size
    weights = tf.get_variable('weights-dft', [filter_size, filter_size,
                                              n_filters_previous_layer, n_filters])
    biases = tf.get_variable('biases-dft', [n_filters])
    output = ricnn.conv_to_circular_shift(output, weights, biases, n_rotations)
    output = tf.nn.relu(output)
    output = ricnn.dft2d_transition(output, n_rotations, batch_size, output_size,
                                    n_filters_previous_layer, n_filters)
    dft_size = ricnn.calculate_dft_output_size(n_filters, n_rotations)

    # Fully-connected layer
    n_nodes = y_size
    weights = tf.get_variable('weights-full', [dft_size, n_nodes])
    biases = tf.get_variable('biases-full', [n_nodes])
    y_output = tf.matmul(output, weights) + biases

    # Define a loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_output))
    # Define an optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run the training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    learning_rate = 0.05
    steps_number = 1000
    # steps_number = 100
    train_data, train_labels = get_data(TRAIN_DATA)
    for i in range(steps_number):
        # Get the next batch
        # input_batch = train_data[i*100:(i+1)*100, :,:,:]
        # labels_batch = train_labels[i*100:(i+1)*100, :]
        input_batch, labels_batch = mnist.train.next_batch(batch_size)
        input_batch = np.reshape(input_batch, (-1,28,28,1))
        feed_dict = {x: input_batch, y: labels_batch}

        # Run the training step
        train_op.run(feed_dict=feed_dict)

        # Print the accuracy progress on the batch every 100 steps

        if i % 100 == 0:
        # if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            print("Step %d, training batch accuracy %g %%" % (i, train_accuracy * 100))

    #Evaluate on the test set
    test_data= mnist.test.images
    test_data = np.reshape(test_data[:100,:], (-1,28,28,1))
    test_label = mnist.test.labels[:100,:]
    test_accuracy = accuracy.eval(feed_dict={x: test_data, y: test_label})
    print("Test accuracy: %g %%" % (test_accuracy * 100))

    # test_data, test_label = get_data(TEST_DATA)
    # test_data = test_data[:100,:,:,:]
    # test_label = test_label[:100,:]
    # test_accuracy = accuracy.eval(feed_dict={x: test_data, y: test_label})
    # print("Test accuracy: %g %%" % (test_accuracy * 100))

