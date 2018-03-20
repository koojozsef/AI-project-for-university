# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:44:59 2018

@author: koojo

@title: K-Nearest neighbor classifyer for labeled images

@description: This file contains a script which can classify images by determining the nearest neighbors.
Image URL-s are from http://image-net.org/index
"""

# open and plot image from example URL: http://farm3.static.flickr.com/2053/2166599298_80c82ca26d.jpg
from PIL import Image
import requests
from io import BytesIO

#read URL-s from file
fname = "..\data\images_URL\imgurl_cat.txt"
with open(fname) as f:
    URL = f.readlines()

#URL = "http://e-vet.com/gallery2/d/57-2/cat.jpg","http://www.wikihow.com/images/thumb/5/58/Feline-eye...-7072.jpg/250px-Feline-eye...-7072.jpg"
#imgArray = []
size = 64,64
i=1
#reads the image from URL
for url in URL:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        #imgArray.append(img)
        
        #save image to ../data/images_gen/cat/
        img.thumbnail(size, Image.ANTIALIAS)
        name = "%04d.jpg" % i
        savepath = "..\data\images_gen\cat\cat_" + name
        img.save(savepath)
        i=i+1
    except:
        print("error")