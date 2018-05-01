# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:44:59 2018

@author: koojo

@title: K-Nearest neighbor classifyer for labeled images

@description: This file contains a script which can classify images by determining the nearest neighbors.

Images are from http://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 python version used for this program
downloaded batch-es shall be located at "../info/cifar-10-batches-py/" directory
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


#read base images
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
    
dictionary = unpickle('../info/cifar-10-batches-py/data_batch_1') #loading first batch
data = dictionary['data']
labels = dictionary['labels']

dictionary = unpickle('../info/cifar-10-batches-py/data_batch_2') #loading first batch
data_test = dictionary['data']
labels_test = dictionary['labels']

""" 
-------------------showing the first data as image-----------------------------

a=data[1]#first image :1024 red, 1024 green, 1024 blue
b=a.reshape(3,32,32).transpose(1,2,0)
img = Image.fromarray(b)
img.show()

-------------------------------------------------------------------------------
"""

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    print X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

nn=NearestNeighbor()

nn.train(data,labels)
estimated = nn.predict(data_test[:10])
real = labels_test[:10]

# show the estimated images with labels
fig=plt.figure(figsize=(16,16))
for i in range(0, 10):
    b=data_test[i].reshape(3,32,32).transpose(1,2,0)
    imgs = Image.fromarray(b)
    fig.add_subplot(1,10,i+1)
    plt.imshow(imgs)
    plt.title("real: " + (str)(real[i]) + "\n est.: " + (str)((int)(estimated[i])))
plt.show()

















