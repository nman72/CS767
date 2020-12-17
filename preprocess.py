#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:26:49 2020

@author: nman
"""

import numpy as np
import pickle
from random import sample

import PIL.Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# This function helps to create a pickle file from all the images
def DataBase_creator(archivezip, nwidth, nheight, save_name):
    # nwidth x nheight = number of features because images have nwidth x nheight pixels
    s = (len(archivezip)-1, nwidth, nheight,3)
    allImage = np.zeros(s)
    for i in range(1,len(archivezip)):
        filename = archivezip[i]
        image = PIL.Image.open(filename) # open colour image
        image = image.resize((nwidth, nheight))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0) # 255 = max of the value of a pixel
        allImage[i-1]=image
    
    # we save the newly created database
    pickle.dump(allImage, open( save_name + '.p', "wb" ), protocol=4 )
    
# Function to create one-hot labels
def matrix_Bin(labels):
    labels_bin=np.array([])

    labels_name, labels0 = np.unique(labels, return_inverse=True)
    #print(labels0)
    
    for _, i in enumerate(np.unique(labels0).astype(int)):
        labels_bin0 = np.where(labels0 == np.unique(labels0)[i], 1., 0.)
        labels_bin0 = labels_bin0.reshape(1,labels_bin0.shape[0])

        if (labels_bin.shape[0] == 0):
            labels_bin = labels_bin0
        else:
            labels_bin = np.concatenate((labels_bin,labels_bin0 ),axis=0)

    #print("Nber SubVariables {0}".format(np.unique(labels0).shape[0]))
    labels_bin = labels_bin.transpose()
    #print("Shape : {0}".format(labels_bin.shape))
    
    return labels_name, labels_bin

def train_test_creation(x, data, toPred): 
  indices = sample(range(data.shape[0]),int(x * data.shape[0])) 
  indices = np.sort(indices, axis=None) 
  
  index = np.arange(data.shape[0]) 
  reverse_index = np.delete(index, indices,0)
  
  train_toUse = data[indices]
  train_toPred = toPred[indices]
  test_toUse = data[reverse_index]
  test_toPred = toPred[reverse_index]

  return train_toUse, train_toPred, test_toUse, test_toPred