from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import zipfile
import tensorflow.python.platform

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import array
import h5py
import shutil

import matplotlib.pyplot as plt;

SOURCE_URL = 'http://www.iro.umontreal.ca/~lisa/icml2007data/'

work_directory = 'mnist'


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def converter(filename):
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        maybe_download(filename)
    zip_ref = zipfile.ZipFile(filepath, 'r')
    subfold = os.path.splitext(filename)[0]
    if os.path.exists(subfold):
        shutil.rmtree(subfold, ignore_errors=True)
        os.mkdir(subfold)
    zip_ref.extractall(subfold)
    zip_ref.close()
    for f in os.listdir(subfold):
        filepath = os.path.join(subfold, f)
        try:
            data = numpy.loadtxt(filepath)
            raw_data = data[:, :-1]
            labels = data[:, -1:]
            labels = numpy.reshape(labels,(-1)).astype('int')
            data = numpy.reshape(raw_data, (-1, 28, 28,1))
        except Exception as e:
            print(e)
        if 'train' in f:
            save_h5py('train', data[: -2000, :, :, :], labels[: -2000])
            save_h5py('validation', data[-2000:, :, :, :], labels[-2000:-1])
        else:
            save_h5py('test', data, labels)
    print('Save data to h5py successfully.')


def save_h5py(output, data, labels):
    output_file = os.path.join(work_directory, output + '.hdf5')
    if os.path.exists(output_file):
        return
    with h5py.File(output_file, 'w') as of:
        of.create_dataset('data', data.shape, dtype='f', data=data)
        of.create_dataset('labels', labels.shape, dtype='f', data=labels)


FILENAME = 'mnist_rotation_new.zip'

local_file = converter(FILENAME)
