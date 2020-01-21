import os
import requests
import zipfile

import scipy
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py
import numpy as np
import math
from sklearn.utils import shuffle



from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import add
from keras.activations import relu, softmax,sigmoid
from keras.models import Model
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,EarlyStopping

filename = 'OralCancer_DataSet3.zip'
cur_dir = '/data2/team16a/'
file = os.path.join(cur_dir, filename)
extract_folder='OralCancer_DataSet3'
data_dir = os.path.join(os.path.join(cur_dir, extract_folder))
class PCA(object):

    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        """
        The computation works as follows:
        The covariance is C = 1/(n-1) * D * D.T
        The eigendecomp of C is: C = V Sigma V.T
        Let Y = 1/sqrt(n-1) * D
        Let U S V = svd(Y),
        Then the columns of U are the eigenvectors of:
        Y * Y.T = C
        And the singular values S are the sqrts of the eigenvalues of C
        We can apply PCA by multiplying by U.T
        """

        # We require scaled, zero-mean data to SVD,
        # But we don't want to copy or modify user data
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=False, ZCA=False,
                  regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by Sigma^(-1/2) U.T
        Any orthogonal transformation of this is also white,
        and when ZCA=True we choose:
         U Sigma^(-1/2) U.T
        """
        if whiten:
            # Compute Sigma^(-1/2) = S^-1,
            # with smoothing for numerical stability
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                # The ZCA whitening matrix
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                # The whitening matrix
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)

def download_data():
    if os.path.exists(file):
        return

    username = 'project'
    password = 'CancerCells'

    url = 'http://www.cb.uu.se/~joakim/OralData3/OralCancer_DataSet3.zip'

    r = requests.get(url, auth=(username, password))

    if r.status_code == 200:
        with open(filename, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)
    else:
        print(r)


def unzip_file():
    if os.path.isdir(data_dir):
        return

    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall(cur_dir)
    zip_ref.close()


def save_h5py(output, data, labels):
    output_file = os.path.join(cur_dir, output + '.hdf5')
    if os.path.exists(output_file):
        return
    with h5py.File(output_file, 'w') as of:
        of.create_dataset('data', data.shape, dtype='float32', data=data)
        of.create_dataset('labels', labels.shape, dtype='int32', data=labels)


def save_npz(output, data,labels):
    output_file = os.path.join(cur_dir, output + '.npz')
    if os.path.exists(output_file):
        return
    np.savez(output_file, data=data, labels=labels)
def read_npz(file_path):
    npzfile = np.load(file_path)
    return npzfile['data'], npzfile['labels']

def save_npy(output, data, labels):
    output_file_data = os.path.join(cur_dir, output + '_data.npy')
    if os.path.exists(output_file_data):
        return
    np.save(output_file_data, data)

    output_file_label = os.path.join(cur_dir, output + '_label.npy')
    if os.path.exists(output_file_label):
        return
    np.save(output_file_label, labels)

def read_npy(output):
    output_file_data = os.path.join(cur_dir, output + '_data.npy')
    output_file_label = os.path.join(cur_dir, output + '_label.npy')
    data = np.load(output_file_data, mmap_mode='r')
    labels = np.load(output_file_label, mmap_mode='r')
    return data, labels

def normalize(data, eps=1e-8):
    mean = data.mean(axis=(1, 2, 3), keepdims=True)
    data -=mean
    std = np.sqrt(data.var(axis=(1, 2, 3), ddof=1, keepdims=True))
    std[std < eps] = 1.
    data /= std
    return data

def white(train_data_all, test_data):

    train_data_all_flat = train_data_all.reshape(train_data_all.shape[0], -1).T
    test_data_flat = test_data.reshape(test_data.shape[0], -1).T
    pca_all = PCA(D=train_data_all_flat, n_components=train_data_all_flat.shape[1])

    print
    '   Whitening data...'
    train_data_all_flat = pca_all.transform(D=train_data_all_flat, whiten=True, ZCA=True)
    train_data_all = train_data_all_flat.T.reshape(train_data_all.shape)
    test_data_flat = pca_all.transform(D=test_data_flat, whiten=True, ZCA=True)
    test_data = test_data_flat.T.reshape(test_data.shape)

    return train_data_all, test_data


def large_data_shuffle():
    train_data,train_labels = read_npy('train')
    test_data, test_labels = read_npy('test')
    train_data, test_data = white(train_data, test_data)
    save_npz('train',train_data,train_labels)
    save_npy('shuffle/train', train_data,train_labels)
    save_npz('test', test_data, test_labels)
    save_npy('shuffle/test', test_data, test_labels)


def read_images(fold, no):
    data = np.zeros(shape=(no, 80, 80,3), dtype='float32')
    label = np.zeros(shape=(no), dtype='int32')
    for subdirs, dirs, files in os.walk(fold):
        i = 0
        for dir in dirs:
            cancer=0
            if dir=='Cancer':
                cancer=1
            subdir = os.path.join(fold, dir)
            for image_file in os.listdir(subdir):
                file_path = os.path.join(subdir, image_file)
                im = imread(file_path)
                im = im.astype(np.float32)
                # gray = np.dot(im, [0.2989, 0.5870, 0.1140])
                # gray = np.expand_dims(gray, axis=-1)
                data[i,:,:,:] = im
                label[i]=cancer
                i=i+1
    return data, label

train_fold=os.path.join(cur_dir,extract_folder, 'train')
test_fold=os.path.join(cur_dir, extract_folder, 'test')
train_no=73303
test_no=55514
train_data, train_label= read_images(train_fold, train_no)
train_data = normalize(train_data)
train_data, train_label= shuffle(train_data, train_label, random_state=0)
# save_npy('train', train_data, train_label)
print('Done with saving training data')
# save_npz('train', train_data, train_label)
# # save_h5py('train', train_data, train_label)
test_data, test_label= read_images(test_fold, test_no)
test_data = normalize(test_data)
test_data, test_label= shuffle(test_data, test_label, random_state=0)

train_data, test_data = white(train_data, test_data)
print('done with whiten')
# # save_h5py('test', test_data, test_label)
# save_npz('test', test_data, test_label)
save_npy('train', train_data, train_label)
save_npy('test', test_data, test_label)
print('Done with saving testing data')
# large_data_shuffle()






