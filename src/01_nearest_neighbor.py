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

URL = "http://farm3.static.flickr.com/2053/2166599298_80c82ca26d.jpg"
imgArray = []

#reads the image from URL 3 times
for i in range (0,3):
    response = requests.get(URL)
    img = Image.open(BytesIO(response.content))
    imgArray.append(img)

#opens the first 3 image from the array
for i in range (0,3):
    imgArray[i].show()