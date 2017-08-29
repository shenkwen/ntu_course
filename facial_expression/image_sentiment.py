# -*- coding:utf-8 -*-
"""
use CNN for image sentiment classification
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import Callback
import keras.backend as K


# ********************** visualization **************************
def show_image(dataset):
    x, y = dataset
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                  4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    fig, axes = plt.subplots(5, 8)
    i = 523
    for ax in axes:
        for sub_ax in ax:
            sub_ax.imshow(x[i], cmap='gray')
            sub_ax.set_title(label_dict[y[i]], size=8)
            sub_ax.axis('off')
            i += 1
    fig.tight_layout()


# ***************************** CNN *****************************
# data prepare
def data_prepare(dataset, num_labels, num_channel=1, sample_size=0):
    x, y = dataset
    num = x.shape[0]
    # sample
    if sample_size != 0:
        idx = np.random.randint(0, num, sample_size)
        x, y = x[idx], y[idx]

    # reshape
    num_case, img_rows, img_cols = x.shape
    if K.image_data_format() == 'channel_first':
        x_channel = x.reshape((num_case, num_channel, img_rows, img_cols))
        input_shape = (num_channel, img_rows, img_cols)
    else:
        x_channel = x.reshape((num_case, img_rows, img_cols, num_channel))
        input_shape = (img_rows, img_cols, num_channel)

    # normalize
    x_std = x_channel / 255
    y_dummy = to_categorical(y, num_labels)

    return x_std, y_dummy, input_shape


# set model and train
def set_convnet(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# set callback
class TrainHistory(Callback):
    def on_train_begin(self, logs={}):
        self.tr_accs = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))


def main():
    os.chdir('D:\\workspace\\data_analysis\\NTU_courses\\ml17_hw3')

    # train, test = file_to_array('fer2013')
    # to_pickle(train, 'train')
    # to_pickle(test, 'test')

    # read from pickle
    train = read_pickle('train')
    test = read_pickle('test')

    # train
    train_x, train_y, input_shape = data_prepare(train, 7)
    test_x, test_y, input_shape = data_prepare(test, 7)

    model = set_convnet(input_shape)
    model.fit(train_x, train_y, epochs=15, batch_size=128,
              validation_split=0.3, callbacks=TrainHistory)
    # validation_data=(test_x, test_y),
