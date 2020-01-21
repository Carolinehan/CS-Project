import argparse
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import models.GCNN as GCNN
import models.cnn as cnn
import models.RiCNN as RiCNN
import shutil
import models.HNets as HNets

VALID='VALID'
best_model='models/best_model'
results_fold='results'
batch_size = 100
import data
cancer_split=50000

def save_results(train_acc, train_loss, val_acc, val_loss, datadir):
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

def save_pred_results(results, datadir):
    result_dir = datadir + '/pred_' + time.strftime('r%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, 'pred_results.csv')
    np.savetxt(result_path, results, delimiter=',')

def save_test_results(results, datadir):
    result_dir = datadir + '/test_' + time.strftime('r%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, 'test_results.txt')
    with open(result_path, 'w+') as f:
        f.write(results)


def plot(model, epochs, train_acc, val_acc):

    plt.plot(epochs, train_acc, 'k', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.ylim(0, 1.1)
    plt.title('Training and validation accuracy')
    plt.xlabel = 'Epochs'
    plt.ylabel = 'Accuracy'
    plt.legend()
    result_dir =results_fold +'/'+model+ '/train_' + time.strftime('r%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, 'plot.png')
    plt.savefig(result_path)
    plt.close()

def get_learning_rate(current, best, counter, learning_rate):
   """If have not seen accuracy improvement in delay epochs, then divide
   learning rate by 10
   """
   delay = 12
   lr_div =10.
   if current > best:
      best = current
      counter = 0
   elif counter > delay:
      learning_rate = learning_rate / lr_div
      counter = 0
   else:
      counter += 1
   return (best, counter, learning_rate)

def get_acc(sess, start, end, data, labels, x, y, accuracy, loss, train_op, pred_results, type, train_phase,learning_rate,lr,model, train=False):
    val_step=0
    loss_value = 0
    val_acc=0
    pred_all = None
    if type == 'Test':
        pred_all=np.zeros(end - start)
    for i in range(start, end, batch_size):
        batch_start = time.time()
        # Get the next batch
        if model == 'RiCNN' or model == 'HNets':
            if i+batch_size > end:
                continue
        input_batch = data[i:i + batch_size, :, :, :]
        labels_batch = labels[i:i + batch_size]

        if train:
            feed_dict = {x: input_batch, y: labels_batch, train_phase: train, learning_rate:lr}
            sess.run(train_op, feed_dict=feed_dict)
        else:
            feed_dict = {x: input_batch, y: labels_batch, train_phase: train}
        loss_percent =sess.run(loss, feed_dict=feed_dict)
        train_accuracy =sess.run(accuracy, feed_dict=feed_dict)
        if type == 'Test':
            pred = pred_results.eval(feed_dict= feed_dict, session = sess)
            pred_all[i:i + batch_size] = pred
        val_step += 1
        val_acc += train_accuracy
        loss_value += loss_percent
        if train:
            batch_time = time.time() - batch_start
            print("Step %d, time %f ms, training batch accuracy %g%%  loss %f" % (
            val_step*batch_size, batch_time * 1000, train_accuracy * 100, loss_percent))
    acc = val_acc / val_step
    loss_value = loss_value / val_step
    print("%s accuracy %g%%, loss %s" % (type, acc * 100, loss_value))
    return acc, loss_value, pred_all

def train(epochs, datadir, model):
    save_model= best_model
<<<<<<< HEAD
    lr =  0.001
    counter =0
=======
    lr =  0.0076
>>>>>>> 32d1be7b2d56ae3596fe1ab88af1518cd4623bb9
    if not os.path.exists(save_model):
        os.mkdir(save_model)
    if 'mnist' in datadir:
        x_size = 28
        x_depth =1
        y_size = 10
        train_data, train_labels, val_data, val_labels = data.get_minist_data(datadir, x_size, x_depth)
        save_model = os.path.join(save_model, 'mnist')
    elif 'cancer' in datadir:
        x_size = 80
        x_depth = 3
        y_size = 2
        train_data,train_labels = data.read_cancer_data('train', datadir)
        val_data, val_labels = train_data,train_labels
        save_model = os.path.join(save_model, 'cancer')
    if not os.path.exists(save_model):
        os.mkdir(save_model)
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, x_size, x_size, x_depth], name='input')
        y = tf.placeholder(dtype=tf.int64, shape=[None])
        train_phase = tf.compat.v1.placeholder(tf.bool, name='train_phase')
        learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        if model == 'cnn':
            logits = cnn.get_model(x,x_depth, y_size, train_phase)
            save_model = os.path.join(save_model, 'cnn')
        elif model == 'RiCNN':

             logits = RiCNN.get_model(x,x_size, x_depth, y_size, train_phase)
             save_model = os.path.join(save_model, 'ricnn')
        elif model == 'GCNN':
            logits = GCNN.get_model(x, x_depth, y_size, train_phase)
            save_model = os.path.join(save_model, 'gcnn')
        elif model == 'HNets':
            logits = HNets.get_model(x,x_size, x_depth, y_size, train_phase)
            save_model = os.path.join(save_model, 'hnets')

        if not os.path.exists(save_model):
            os.mkdir(save_model)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optim.compute_gradients(loss)
        modified_gvs = []
        # We precondition the phases, for faster descent, in the same way as biases
        for g, v in grads_and_vars:
            if 'psi' in v.name:
                g = 7.8 * g
            modified_gvs.append((g, v))
        train_op = optim.apply_gradients(modified_gvs)
        correct_prediction = tf.equal(y, tf.argmax(logits, 1))
        pred_results = tf.argmax(logits,1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        best_acc = 0
        sess = tf.InteractiveSession()

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()], feed_dict={train_phase: True})
        total_start_time = time.time()
        train_accs=np.zeros(epochs)
        val_accs = np.zeros(epochs)
        train_losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        saved_model_path = os.path.join(save_model, 'best.ckpt')
        for epoch in range(epochs):
            print('start training epoch %g' % (epoch+1))
            epoch_start=time.time()
            if 'mnist' in datadir:
                start =0
                end = len(train_data)
            elif 'cancer' in datadir:
                start = 0
                end = cancer_split
            train_acc, train_loss,_ = get_acc(sess, start, end, train_data, train_labels, x, y, accuracy, loss, train_op, pred_results, 'Training',train_phase,learning_rate,lr, model, True)
            epoch_time = time.time() - epoch_start
            print('epoch %g training time %f s' % (epoch+1, epoch_time))
            train_accs[epoch] = train_acc
            train_losses[epoch] = train_loss
            if 'mnist' in datadir:
                start =0
                end = len(val_data)
            elif 'cancer' in datadir:
                start = cancer_split
                end = len(val_data)
            val_acc, val_loss,_ = get_acc(sess, start, end, val_data, val_labels, x, y, accuracy, loss, train_op,pred_results,'Validation',train_phase,learning_rate,lr, model)
            val_accs[epoch] = val_acc
            val_losses[epoch] = val_loss

            best, counter, lr = get_learning_rate(val_acc, best_acc, counter, lr)

            if val_acc > best_acc:
                best_acc = val_acc
                if os.path.exists(save_model):
                    shutil.rmtree(save_model)
                os.mkdir(save_model)
                saver.save(sess, saved_model_path)
<<<<<<< HEAD

        if 'mnist' in datadir:
            test_data, test_labels = data.get_test_data(datadir, x_size, x_depth)
        elif 'cancer' in datadir:
            test_data, test_labels =data.read_cancer_data('test', datadir)
        start = 0
        end = len(test_data)
=======
            lr = lr * np.power(0.1, epoch / 50)
    if 'mnist' in datadir:
        test_data, test_labels = data.get_test_data(datadir, x_size, x_depth)
    elif 'cancer' in datadir:
        test_data, test_labels =data.read_cancer_data('test', datadir)
    start = 0
    end = len(test_data)

    ckpt = tf.train.get_checkpoint_state(save_model)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
    test_acc, test_loss, pred = get_acc(sess, start, end,test_data, test_labels, x, y, accuracy, loss, train_op,pred_results,'Test', train_phase,learning_rate,lr, model)
    total_time=time.time()-total_start_time
    print('Total time: %fs' %total_time)
    save_results(train_accs, train_losses, val_accs, val_losses,datadir)
    plot(model, range(epochs), train_accs, val_accs)
>>>>>>> 32d1be7b2d56ae3596fe1ab88af1518cd4623bb9

        ckpt = tf.train.get_checkpoint_state(save_model)
        saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
        test_acc, test_loss, pred = get_acc(sess, start, end,test_data, test_labels, x, y, accuracy, loss, train_op,pred_results,'Test', train_phase,learning_rate,lr, model)
        total_time=time.time()-total_start_time
        print('Total time: %fs' %total_time)
        save_results(train_accs, train_losses, val_accs, val_losses,datadir)
        plot(model, range(epochs), train_accs, val_accs)

        save_pred_results(pred, os.path.join(results_fold, model))
        save_test_results("Test accuracy %g%%, loss %s datadir %s  total time %f" % (test_acc * 100, test_loss, datadir, total_time), os.path.join(results_fold, model))
        sess.close()

if __name__ == '__main__':


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--datadir', type=str, default='mnist-rot')
    # parser.add_argument('--model', type=str, default='cnn')
    # parser.add_argument('--epochs', type=int, default=100)
    #
    # args = parser.parse_args()
    # train(**vars(args))
    # train(100, 'mnist-rot', 'RiCNN')
<<<<<<< HEAD
    train_epoch = 100
    train(train_epoch, 'mnist-rot', 'RiCNN')
    train(train_epoch, 'mnist-rot', 'cnn')
    train(train_epoch, 'mnist-rot', 'GCNN')
    train(train_epoch, 'mnist-rot', 'HNets')
    #
    train_epoch = 100
    train(train_epoch, 'oral-cancer', 'GCNN')
    train(train_epoch, 'oral-cancer', 'RiCNN')
    train(train_epoch, 'oral-cancer', 'cnn')
    train(train_epoch, 'oral-cancer', 'HNets')
    # train(100, 'oral-cancer', 'HNets')
=======

    # train(100, 'mnist-rot', 'RiCNN')
    # train(100, 'mnist-rot', 'cnn')
    # train(100, 'mnist-rot', 'GCNN')
    # train(100, 'mnist-rot', 'HNets')
    #
    # train(100, 'oral-cancer', 'RiCNN')
    # train(100, 'oral-cancer', 'GCNN')
    # train(100, 'oral-cancer', 'cnn')
   # train(50, 'oral-cancer', 'HNets')
    train(100, 'oral-cancer', 'HNets')
>>>>>>> 32d1be7b2d56ae3596fe1ab88af1518cd4623bb9

