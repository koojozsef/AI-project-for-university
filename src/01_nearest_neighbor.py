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


#read base images
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
    
dictionary = unpickle('../info/cifar-10-batches-py/data_batch_1') #loading first batch
data = dictionary['data']
a=data[1]#first image :1024 red, 1024 green, 1024 blue
b=a.reshape(3,32,32).transpose(1,2,0)
img = Image.fromarray(b)
img.show()

#define class base values

#read test images

#compute values

#classification