from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D


data_folder = './data'
dlog = pd.read_csv('%s/driving_log.csv' % data_folder)

images = [plt.imread('%s/%s' % (data_folder, img)) for img in dlog.center]

X_orig = np.array(images)
y_orig = np.array(dlog.steering, dtype=float)


def data_augmentation(X, y):
    """ Flip each picture horizontally """
    X_aug = X[:, :, ::-1, :]
    y_aug = - y
    return np.concatenate((X, X_aug), axis=0), \
        np.concatenate((y, y_aug), axis=0)


X_train, y_train = data_augmentation(X_orig, y_orig)


def NVIDIA():

    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.75))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


sys.exit(0)

model = NVIDIA()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=8)

model.save('model.h5')
