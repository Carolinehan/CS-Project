import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import ricnn

VALID='VALID'
channels = 10
best_model='best_model'
results_fold='results'
datadir ='../mnist-rot'
trainfn='train.npz'
valfn='valid.npz'
testfn='test.npz'


filter_size = 3
n_filters = 128
n_rotations = 4
stride = 1
pool_stride = 2
def build_layer(layer_order, previous_layer, weight_size=n_filters, activation='relu'):
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

def get_normal_model(x):
    l2 = build_layer(1, x, 128)
    l3= tf.layers.dropout(l2,0.2)
    images_flat = tf.contrib.layers.flatten(l3)
    l5 = tf.contrib.layers.fully_connected(images_flat, y_size, tf.nn.softmax)
    return l5
def get_model(x):
    l1 = build_layer(1,x)
    l2 = build_layer(2,l1)
    l2 = tf.nn.max_pool(l2, [1, pool_stride, pool_stride, 1],
                                [1, pool_stride, pool_stride, 1], padding='VALID')

    l3 = build_layer(3,l2)
    l4 = build_layer(4,l3)
    l5 = build_layer(5,l4)
    l6 = build_layer(6,l5)
    images_flat = tf.contrib.layers.flatten(l6)
    l7 = tf.contrib.layers.fully_connected(images_flat, y_size, tf.nn.softmax)
    return l7

def preprocess_mnist_data(train_data, test_data, train_labels, test_labels):
    # train_mean = np.mean(train_data)  # compute mean over all pixels make sure equivariance is preserved
    # train_data -= train_mean
    # test_data -= train_mean
    # train_std = np.std(train_data)
    # train_data /= train_std
    # test_data /= train_std
    # train_data = train_data.astype(np.float32)
    # test_data = test_data.astype(np.float32)
    # train_labels = train_labels.astype(np.int32)
    # test_labels = test_labels.astype(np.int32)

    return train_data, test_data, train_labels, test_labels
def save_results(train_acc, train_loss, val_acc, val_loss):
    # if args.restart_from is None:
    result_dir =datadir + '/' + time.strftime('r%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    train_acc_path = os.path.join(result_dir, 'train_acc.csv')
    train_loss_path = os.path.join(result_dir, 'train_loss.csv')
    val_acc_path = os.path.join(result_dir, 'val_acc.csv')
    val_loss_path = os.path.join(result_dir, 'val_loss.csv')
    np.savetxt(train_acc_path, train_acc, delimiter=',')
    np.savetxt(train_loss_path, train_loss, delimiter=',')
    np.savetxt(val_acc_path, val_acc, delimiter=',')
    np.savetxt(val_loss_path, val_loss, delimiter=',')

def plot(epochs, train_acc, val_acc):
    plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('CNN Traing and validation accuracy')
    plt.xlabel = 'Epochs'
    plt.ylabel = 'Accuracy'
    plt.legend()
    plt.show()

def read_data(file_name):
    set = np.load(os.path.join(datadir, file_name))
    data = set['data']
    data = np.reshape(data, (-1, x_size, x_size, x_depth))
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

def get_acc(data, labels, x, y, accuracy, loss, train_op, type, train=False):
    val_step=0
    loss_value = 0
    val_acc=0
    for i in range(0, len(data), batch_size):
        batch_start = time.time()
        # Get the next batch
        input_batch = data[i:i + batch_size, :, :, :]
        labels_batch = labels[i:i + batch_size]
        feed_dict = {x: input_batch, y: labels_batch}
        if train:
            train_op.run(feed_dict=feed_dict)
        loss_percent = loss.eval(feed_dict=feed_dict)
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        val_step += 1
        val_acc += train_accuracy
        loss_value += loss_percent
        if train:
            batch_time = time.time() - batch_start
            print("Step %d, time %f ms, training batch accuracy %g  loss %f" % (
            i, batch_time * 1000, train_accuracy * 100, loss_percent))
    acc = val_acc / val_step
    loss_value = loss_value / val_step
    print("%s accuracy %g, loss %s" % (type, acc * 100, loss_value))
    return acc, loss_value

def train(epochs):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = get_data()
    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, shape=[None, x_size, x_size, x_depth], name='input')
        y = tf.placeholder(dtype=tf.int64, shape=[None])
        logits = get_model(x)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        correct_prediction = tf.equal(y, tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        best_acc = 0
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        total_start_time = time.time()
        train_accs=np.zeros(epochs)
        val_accs = np.zeros(epochs)
        train_losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        for epoch in range(epochs):
            print('start training epoch %g' % (epoch))
            train_acc, train_loss = get_acc(train_data, train_labels, x, y, accuracy, loss, train_op, 'Training', True)
            train_accs[epoch] = train_acc
            train_losses[epoch] = train_loss
            val_acc, val_loss = get_acc(val_data, val_labels, x, y, accuracy, loss, train_op, 'Validation')
            val_accs[epoch] = val_acc
            val_losses[epoch] = val_loss

    get_acc(test_data, test_labels, x, y, accuracy, loss, train_op,'Test')
    total_time=time.time()-total_start_time
    print('Total time: %fs' %total_time)
    save_results(train_accs, train_losses, val_accs, val_losses)
    plot(range(epochs), train_accs, val_accs)
def restore_model():
    tf.reset_default_graph()
    with tf.session() as sess:
        tf.saved_model.loader.load(sess, ['serve'], best_model)
        sess.run('output:0', feed_dict={})

train(100)
