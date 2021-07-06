import numpy as np
import torch
from torchvision import datasets, transforms

def get_letters():
    data=np.float64(np.load('../data/letters_data.npy'))
    print(data.shape)
    data=np.float32(data)/255.
    train_dat=data[0:90000].reshape((-1,1,28,28))
    val_dat=data[90000:107000].reshape((-1,1,28,28))
    test_dat=data[107000:124000].reshape((-1,1,28,28))
    return (train_dat, None), (val_dat, None), (test_dat, None)

def get_mnist():
    data=np.float64(np.load('../data/MNIST.npy'))
    labels=np.float32(np.load('../data/MNIST_labels.npy'))
    print(data.shape)
    data=np.float32(data)/255.
    train_dat=data[0:50000].reshape((-1,1,28,28))
    train_labels=np.int32(labels[0:50000])
    val_dat=data[50000:60000].reshape((-1,1,28,28))
    val_labels=np.int32(labels[50000:60000])
    test_dat=data[60000:70000].reshape((-1,1,28,28))
    test_labels=np.int32(labels[60000:70000])
    return (train_dat, train_labels), (val_dat, val_labels), (test_dat, test_labels)

def get_mnist_tr():
    data=np.float64(np.load('../data/MNIST_TRANSFORM_data.npy'))
    labels=np.float32(np.load('../data/MNIST_labels.npy'))
    print(data.shape)
    data=np.float32(data)/255.
    train_dat=data[0:50000].reshape((-1,1,28,28))
    train_labels=np.int32(labels[0:50000])
    val_dat=data[50000:60000].reshape((-1,1,28,28))
    val_labels=np.int32(labels[50000:60000])
    test_dat=data[60000:70000].reshape((-1,1,28,28))
    test_labels=np.int32(labels[60000:70000])
    return (train_dat, train_labels), (val_dat, val_labels), (test_dat, test_labels)

def get_cifar10():
    train = datasets.CIFAR10('../data', train=True, download=True)
    test = datasets.CIFAR10('../data', train=False, download=True)
    train_dat = (np.float64(train.data.squeeze()[0:46000]).transpose(0,3,1,2)/255. - 0.5)/0.5
    train_labels = np.float32(train.targets[0:46000])
    val_dat = (np.float64(train.data.squeeze()[46000:]).transpose(0,3,1,2)/255. - 0.5)/0.5
    val_labels = np.float32(train.targets[46000:])
    test_dat = (np.float64(test.data.squeeze()).transpose(0,3,1,2)/255. - 0.5)/0.5
    test_labels = np.float32(test.targets)
    return (train_dat, train_labels), (val_dat, val_labels), (test_dat, test_labels)

def get_cifar100():
    train = datasets.CIFAR100('../data', train=True, download=True)
    test = datasets.CIFAR100('../data', train=False, download=True)
    train_dat = (np.float64(train.data.squeeze()[0:46000]).transpose(0,3,1,2)/255. - 0.5)/0.5
    train_labels = np.float32(train.targets[0:46000])
    val_dat = (np.float64(train.data.squeeze()[46000:]).transpose(0,3,1,2)/255. - 0.5)/0.5
    val_labels = np.float32(train.targets[46000:])
    test_dat = (np.float64(test.data.squeeze()).transpose(0,3,1,2)/255. - 0.5)/0.5
    test_labels = np.float32(test.targets)
    return (train_dat, train_labels), (val_dat, val_labels), (test_dat, test_labels)

def get_data(data_set):
    if (data_set=="mnist"):
        return(get_mnist())
    if (data_set=="mnist_tr"):
        return(get_mnist_tr())
    if (data_set=="letters"):
        return(get_letters())
    if (data_set=="cifar10"):
        return(get_cifar10())
    if (data_set=="cifar100"):
        return(get_cifar100())
    else:
        raise ValueError('No dataset named {}'.format(data_set))
    