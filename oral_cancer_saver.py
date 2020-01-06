import os
import requests
import zipfile

import scipy
from matplotlib.image import imread
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
cur_dir = '/home/uppsala/project/data/'
file = os.path.join(cur_dir, filename)
extract_folder='OralCancer_DataSet3'
data_dir = os.path.join(os.path.join(cur_dir, extract_folder))


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

def normalise(data):
    data = data.astype(np.float32)
    mean = np.mean(data)  # compute mean over all pixels make sure equivariance is preserved
    data -= mean

    std = np.std(data)
    data /= std
    return data

def large_data_shuffle(output):
    data,labels = read_npy(output)
    data = normalise(data)
    n_data, n_label = shuffle(data, labels,random_state=0)
    save_npz(output, n_data, n_label)
    save_npy('shuffle/'+output, n_data, n_label)

def read_images(fold, no):
    data = np.zeros(shape=(no, 80, 80, 3), dtype='float32')
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
                data[i,:,:,:] = im
                label[i]=cancer
                i=i+1
    return data, label

train_fold=os.path.join(cur_dir,extract_folder, 'train')
test_fold=os.path.join(cur_dir, extract_folder, 'test')
train_no=73303
test_no=55514
train_data, train_label= read_images(train_fold, train_no)
save_npy('train', train_data, train_label)
# print('Done with saving training data')
# # save_npz('train', train_data, train_label)
# # save_h5py('train', train_data, train_label)
test_data, test_label= read_images(test_fold, test_no)
# # save_h5py('test', test_data, test_label)
# # save_npz('test', test_data, test_label)
save_npy('test', test_data, test_label)
# print('Done with saving testing data')
large_data_shuffle('train')
large_data_shuffle('test')





