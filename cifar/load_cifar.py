import os, cPickle
from numpy import *
import util

N_per_batch = 10000
width = 32
height = 32
channels = 3

def load_cifar(path = './data/',
               n_train = None,
               n_test = None,
               grayscale=False,
               shuffle=False):
    """
    Load CIFAR-10 dataset.

    Parameters
    ----------
    path : string, optional
        Path containing CIFAR-10 dumps. By default looks for ./data/
        directory in the current path.
    n_train : int, optional
        Number of training examples. Defaults to 60,000
    n_test : int, optional
        Number of test examples. Defaults to 10,000.
    grayscale : bool, optional
        If true, grayscale all images. False by default.
    shuffle : bool, optional
        If true, shuffles the dataset before splitting. False by default.

    Returns
    -------
    out : dict
        Dictionay containing train and test data, along with labels.
    """
    files = os.listdir(path)
    out = dict()

    def _load(file):
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    if n_train is None: n_train = N_per_batch*(len(files) - 1)
    if n_test is None: n_test = max(N_per_batch,
                                    N_per_batch*len(files) - n_train)

    N = min(N_per_batch*len(files), n_train + n_test)

    n_channels = 1 if grayscale else channels
    data = zeros((N, width*height*n_channels))
    labels = zeros(N)
    n = 0
    for file in files:
        if n >= N: break
        currN = min(N-n, N_per_batch)
        dct = _load(os.path.join(path, file))
        n1 = n+N_per_batch
        dat = dct['data'][0:currN]/255.0
        dat = dat.reshape(currN, channels, width, height)
        dat = dat.swapaxes(1, 3)
        dat = dat.swapaxes(1, 2)
        if grayscale:
            gray_vals = array([0.299, 0.587, 0.114])
            dat = dot(dat, gray_vals).reshape(currN, -1)
        else:
            dat = dat.reshape(currN, -1)
        data[n:n1] = dat
        labels[n:n1] = dct['labels'][0:currN]
        n += N_per_batch
    if shuffle:
        random.shuffle(data)

    train_data = data[:n_train]
    test_data = data[n_train:n_train+n_test]
    train_labels = labels[:n_train]
    test_labels = labels[n_train:n_train+n_test]

    out['train_data'] = train_data
    out['train_labels'] = train_labels
    out['test_data'] = test_data
    out['test_labels'] = test_labels

    return out
