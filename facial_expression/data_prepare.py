# -*- coding:utf-8 -*-

import os
import pickle
import tarfile
import numpy as np


WORK_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORK_PATH, 'data')  # dir containing all data


def maybe_extract(compressed_file_name, force=False):
    """extract compressed file
    """

    # compressed file
    tar_dir = os.path.join(DATA_PATH, compressed_file_name)  # .tar file
    extract_dir = os.path.splitext(tar_dir)[0]  # extracted file

    # extracting file
    if os.path.exists(extract_dir) and not force:
        print('Extracting: %s already exists - Skipping extraction of %s' % (extract_dir, tar_dir))
    else:
        print('Extracting: extract data, this may take a while.')
        tar = tarfile.open(tar_dir)
        tar.extractall(path=DATA_PATH)
        tar.close()

    return extract_dir


def file_to_array(data_dir, data_name):
    """transform csv file to numpy arrays
    """

    # check data source dir
    if not os.path.exists(data_dir):
        raise Exception('extracted dir not existing. try to extract data again')

    data_file = os.path.join(data_dir, data_name)
    if not os.path.exists(data_file):
        raise Exception('%s not exists in %s, ensure the data file name' % (data_name, data_dir))

    # file to list
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    with open(data_file) as f:
        for line in f.readlines():
            s = line.rstrip().split(',')
            if s[-1] == 'Training':
                train_x.append([int(j) for j in s[1].split(' ')])
                train_y.append(int(s[0]))
            elif s[-1] == 'PublicTest':
                test_x.append([int(j) for j in s[1].split(' ')])
                test_y.append(int(s[0]))

    # list to ndarray
    train_x = np.array(train_x)
    train_x = train_x.reshape(train_x.shape[0], 48, 48)
    train_y = np.array(train_y)
    train = [train_x, train_y]

    test_x = np.array(test_x)
    test_x = test_x.reshape(test_x.shape[0], 48, 48)
    test_y = np.array(test_y)
    test = [test_x, test_y]

    return train, test


def load_data():
    """load data: read data from pickle files if existing,
       or read from original compressed file and save to pickle files
    """

    compressed_file_name = 'fer2013.tar'
    data_name = 'fer2013.csv'

    # load data: read data from pickle files if existing, or read from compressed file and save to pickle file
    train_pickle = os.path.join(DATA_PATH, 'train.pickle')
    test_pickle = os.path.join(DATA_PATH, 'test.pickle')

    if not (os.path.exists(train_pickle) and os.path.exists(test_pickle)):  # pickle files not present
        print('pickle files not present - load data from scratch')
        data_dir = maybe_extract(compressed_file_name)  # extract file
        train, test = file_to_array(data_dir, data_name)  # transform to numpy array
        # pickling
        print('Pickling, this may take a while...')
        with open(train_pickle, 'wb') as f:
            pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
        with open(test_pickle, 'wb') as f:
            pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('pickle files already present - loading data from pickle file')
        with open(train_pickle, 'rb') as f:
            train = pickle.load(f)
        with open(test_pickle, 'rb') as f:
            test = pickle.load(f)

    return train, test
