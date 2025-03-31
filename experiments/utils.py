import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split
import os
import gzip
from torchvision import datasets, transforms

def load_data(dataset_name):

    # Code adapted from https://github.com/zahraatashgahi/NeuroFS/blob/main/code/utils.py

    if dataset_name == "coil20":
        # from https://jundongl.github.io/scikit-feature/datasets.html
        mat = scipy.io.loadmat('data/COIL20.mat')
        X = mat['X']
        y = mat['Y'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        y_test = y_test.ravel()
        y_train = y_train.ravel()
        
    elif dataset_name == "USPS":
        # from https://jundongl.github.io/scikit-feature/datasets.html
        mat = scipy.io.loadmat('data/USPS.mat')
        X = mat['X']
        y = mat['Y'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        y_test = y_test.ravel()
        y_train = y_train.ravel()
    
    elif dataset_name == "MNIST":
        # from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

        # Load the data
        with np.load('data/mnist/mnist.npz') as f:
            X_train, y_train = f['x_train'], f['y_train']
            X_test, y_test = f['x_test'], f['y_test']

        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')  
        
    elif dataset_name == "Fashion-MNIST":
        # from https://github.com/zalandoresearch/fashion-mnist

        def load_mnist(path, kind='train'):
            # code from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py

            """Load MNIST data from `path`"""
            labels_path = os.path.join(path,
                                    '%s-labels-idx1-ubyte.gz'
                                    % kind)
            images_path = os.path.join(path,
                                    '%s-images-idx3-ubyte.gz'
                                    % kind)

            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                    offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                    offset=16).reshape(len(labels), 784)

            return images, labels

        X_train, y_train = load_mnist('data/fashion_mnist', kind='train')
        X_test, y_test = load_mnist('data/fashion_mnist', kind='t10k')
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')
    
    elif dataset_name=="isolet":

        # from https://archive.ics.uci.edu/dataset/54/isolet

        x_train = np.genfromtxt('data/isolet/isolet1234.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
        y_train = np.genfromtxt('data/isolet/isolet1234.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')
        x_test = np.genfromtxt('data/isolet/isolet5.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
        y_test = np.genfromtxt('data/isolet/isolet5.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')

        X = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')

    elif dataset_name=="har":         
        # from https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
        X_train = np.loadtxt('data/UCI HAR Dataset/train/X_train.txt')
        y_train = np.loadtxt('data/UCI HAR Dataset/train/y_train.txt')
        X_test =  np.loadtxt('data/UCI HAR Dataset/test/X_test.txt')
        y_test =  np.loadtxt('data/UCI HAR Dataset/test/y_test.txt')
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')

    elif dataset_name == "Prostate_GE":
        # from https://jundongl.github.io/scikit-feature/datasets.html
        mat = scipy.io.loadmat('data/Prostate_GE.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    elif dataset_name=="SMK":
        # from https://jundongl.github.io/scikit-feature/datasets.html
        mat = scipy.io.loadmat('data/SMK_CAN_187.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.MinMaxScaler().fit(X_train)

    elif dataset_name == "GLA-BRA-180":
        # downloaded from https://jundongl.github.io/scikit-feature/OLD/datasets_old.html
        mat = scipy.io.loadmat('data/GLA-BRA-180.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    elif dataset_name == "BASEHOCK":
        # from https://jundongl.github.io/scikit-feature/datasets.html
        mat = scipy.io.loadmat('data/BASEHOCK.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  
        
    elif dataset_name == "arcene":
        # from https://jundongl.github.io/scikit-feature/datasets.html
        mat = scipy.io.loadmat('data/arcene.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    elif dataset_name == "CIFAR-10":
        # from https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(train=True, transform=transform)
        test_dataset = datasets.CIFAR10(train=False, transform=transform)
        X_train = np.array([np.array(img).flatten() for img, _ in train_dataset])
        y_train = np.array([label for _, label in train_dataset])
        X_test = np.array([np.array(img).flatten() for img, _ in test_dataset])
        y_test = np.array([label for _, label in test_dataset])

    elif dataset_name == "CIFAR-100":
        # from https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(train=True, transform=transform)
        test_dataset = datasets.CIFAR100(train=False, transform=transform)
        X_train = np.array([np.array(img).flatten() for img, _ in train_dataset])
        y_train = np.array([label for _, label in train_dataset])
        X_test = np.array([np.array(img).flatten() for img, _ in test_dataset])
        y_test = np.array([label for _, label in test_dataset])

    if dataset_name == "BASEHOCK":
        X = preprocessing.StandardScaler().fit_transform(np.concatenate((X_train, X_test)))
        X_train = X[: y_train.shape[0]]
        X_test = X[y_train.shape[0]:]
    else:
        X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((X_train, X_test)))
        X_train = X[: y_train.shape[0]]
        X_test = X[y_train.shape[0]:]
    
    if dataset_name in ["har", "isolet"]:
        y_train = y_train - 1
        y_test = y_test - 1
                
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32') 
    y_train = y_train.astype('int')
    y_test  = y_test.astype('int')
    
    y_train = np.asarray(pd.get_dummies(y_train))
    y_test = np.asarray(pd.get_dummies(y_test))

    y_train = np.array([np.where(r==1)[0][0] for r in y_train])
    y_test = np.array([np.where(r==1)[0][0] for r in y_test])

    return (X_train, y_train), (X_test, y_test)

