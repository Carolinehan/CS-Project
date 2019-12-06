import tensorflow as tf
import numpy as np
import h5py
import os
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linspace

WORK_DICTIONARY ='mnist/'
TRAIN_DATA='train.hdf5'
VALIDATION_DATA='validation.hdf5'
TEST_DATA='test.hdf5'


f = h5py.File(os.path.join(WORK_DICTIONARY, TRAIN_DATA), 'r')
data = np.reshape(f['data'],(-1,28,28))
labels =np.array(f['labels']).astype('int')

fig, axes = plt.subplots(5, 2)
xs = linspace(0, 1, 100)
i = 0
for axis in axes:
    img = data[i,:,:]
    im = Image.fromarray(img)
    axis.plot(xs, xs ** 2)
    axis.imshow(im, extent=(0.4, 0.6, .5, .7), zorder=-1, aspect='auto')
    i = i+1
plt.show()