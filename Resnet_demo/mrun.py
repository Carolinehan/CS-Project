
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import requests
import zipfile
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import add
from keras.activations import relu, softmax,sigmoid
from keras.models import Model
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,EarlyStopping
import keras
import matplotlib.pyplot as plt
import time
import data

results_fold='results'
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

def save_results(train_acc, train_loss, val_acc, val_loss):
    # if args.restart_from is None:
    result_dir =results_fold + '/values/' + time.strftime('r%Y_%m_%d_%H_%M_%S')
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

def save_test_results(results,model):
    result_dir = results_fold + '/'+model+'/test_' + time.strftime('r%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, 'test_results.txt')
    with open(result_path, 'w+') as f:
        f.write(results)

def train(datadir, split, modelname):
    resnet = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(28,28,1), classes=1)
    for layer in resnet.layers:
        layer.trainable = True
    es = EarlyStopping(monitor='val_acc', patience=15)

    x = resnet.output
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    # average pooling across the channels
    # 28x28x48 -> 1x48
    x = GlobalAveragePooling2D()(x)

    # dropout for more robust learning
    x = Dropout(0.2)(x)

    # last softmax layer
    x = Dense(units=1, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation(sigmoid)(x)
    # x=keras.layers.Flatten()(resnet.output)
    # output = Dense(1, activation='softmax', name='fc1000')(x)
    model =Model(input=resnet.input, output= x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    best_model = 'weights.best.'+modelname+'.keras'
    mc = ModelCheckpoint(best_model, monitor='val_acc', save_best_only=True)

    file_name='train'
    # train_data = np.load(os.path.join(datadir, file_name + '_data.npy'), mmap_mode='r')
    # train_labels = np.load(os.path.join(datadir, file_name + '_label.npy'), mmap_mode='r')
    train_data, train_labels, val_data, val_labels = data.get_minist_data(datadir, 28, 1)
    history= model.fit(x=train_data, y=train_labels,verbose=1, batch_size=100,epochs=100, callbacks=[es,mc], validation_data=(val_data, val_labels) )
    history_dict = history.history
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    save_results(acc_values, loss_values, val_acc_values, val_loss_values)
    #plot(modelname,range(100), acc_values, val_acc_values)

    file_name = 'test'
    test_data, test_labels = data.get_test_data(datadir, 28, 1)
    model.load_weights(best_model)
    results = model.evaluate(test_data, test_labels)
    print(results)
    save_test_results(str(results),modelname)


# datadir='/data2/team16a/'
#train(datadir, 0.32, 'avg50000')
#train(datadir, 0.6, 'avg30000')
#train(datadir, 0.86, 'avg10000')


datadir='/home/uppsala/CS-Project/gcnn_experiments/mnist-rot/'
#train(datadir, 0.32, 'per50000')
train(datadir, 0.6, 'mnist')
