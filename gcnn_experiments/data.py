import argparse
import os
import numpy as np
from sklearn.utils import shuffle
import math
import tensorflow as tf
trainfn='train'
valfn='valid'
testfn='test'

def preprocess_mnist_data(train_data, test_data, train_labels, test_labels):
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_mean = np.mean(train_data)  # compute mean over all pixels make sure equivariance is preserved
    train_data -= train_mean
    test_data -= train_mean
    train_std = np.std(train_data)
    train_data /= train_std
    test_data /= train_std

    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    return train_data, test_data, train_labels, test_labels

def normalise_data(data):
    mean = np.mean(data)
    data -= mean
    std = np.std(data)
    data /= std
    data = data.astype(np.float32)
    return data

def read_minst_data(file_name, datadir, x_size, x_depth):
    set = np.load(os.path.join(datadir, file_name+'.npz'), mmap_mode='r')
    data = set['data']
    expected_shape = (-1, x_size, x_size, x_depth)
    if data.shape[1:] != expected_shape[1:]:
        data = np.reshape(data, (-1, x_size, x_size, x_depth))
    labels = set['labels']
    # data, labels = shuffle(data, labels, random_state=0)
    return data, labels

def read_cancer_data(file_name, datadir):
    data = np.load(os.path.join(datadir, file_name+'_data.npy'), mmap_mode='r')
    labels =np.load(os.path.join(datadir, file_name+'_label.npy'), mmap_mode='r')
    return data, labels

def split_data(data, label, interval):
    train_data = data[:interval, :,:,:]
    train_label = label[:interval]
    val_data = data[interval:,:,:,:]
    val_label = label[interval:]
    return train_data, train_label, val_data, val_label


def get_minist_data(datadir, x_size, x_depth):
    train_data, train_labels = read_minst_data(trainfn, datadir, x_size, x_depth)
    interval = 10000
    train_data, train_labels, val_data, val_labels = split_data(train_data, train_labels, interval)
    train_data, val_data, train_labels, val_labels = preprocess_mnist_data(
        train_data, val_data, train_labels, val_labels)
    return train_data,train_labels, val_data, val_labels


def get_test_data(datadir, x_size, x_depth):
    test_data, test_labels = read_minst_data(testfn, datadir, x_size, x_depth)
    test_data= normalise_data(test_data)
    return test_data, test_labels
