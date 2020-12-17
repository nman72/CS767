#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:46:01 2020

@author: nman
"""

from pathlib import Path

import numpy as np
import pickle

import pandas as pd  
import cv2
import sys

from sklearn.model_selection import train_test_split

from glob import glob
from tensorflow import keras
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import cnn
import preprocess
import yolo

import torch
import torchvision.transforms as transforms
import torchvision.models as models

# define model
model = cnn.define_model()
#print(model)
    
dog_train = np.array(glob("dog-breed-identification/train/*"))
dog_test = np.array(glob("dog-breed-identification/test/*"))

print('There are %d total train dog images.' % len(dog_train))
print('There are %d total test dog images.' % len(dog_test))
#print(dog_train.shape)

dog_train_short = dog_train[:100]
dog_test_short = dog_test[:100]
#print(dog_train_short[0])

image_resize = 120

# check if CUDA is available
use_cuda = torch.cuda.is_available()
#print("cuda available? {0}".format(use_cuda)

def load_image(img_path):    
    image = Image.open(img_path)

    # resize to (120, 120)
    in_transform = transforms.Compose([
                        transforms.Resize(size=(image_resize, image_resize)),
                        transforms.ToTensor()])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image).unsqueeze(0)
    return image

def VGG16_predict(img_path):
    # define VGG16 model
    vgg = models.vgg16(pretrained=True)
    img = load_image(img_path)
    if use_cuda:
        img = img.cuda()
    ret = vgg(img)
    #print(ret)
    return torch.max(ret, 1)[1].item() # predicted class index


# predict dog using ImageNet class
#print(VGG16_predict(dog_train_short[0]))

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    pred = VGG16_predict(img_path)
    return pred >= 151 and pred <= 268 # true/false

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

model_file = "final_model.h5"
saved_model = Path(model_file)
age_saved_model = Path("age_final_model.h5")

# Resize the images to 60x60 pixels to keep the pickle files under 1GB in size
train_file = Path("train.p")
if (not train_file.is_file()):
    preprocess.DataBase_creator(archivezip = dog_train, nwidth = image_resize, nheight = image_resize , save_name = "train")
test_file = Path("test.p")
if (not test_file.is_file()):
    preprocess.DataBase_creator(archivezip = dog_test, nwidth = image_resize, nheight = image_resize , save_name = "test")
    
# load TRAIN
train = pickle.load( open( "train.p", "rb" ), encoding='bytes')
#print(train.shape)

# load TEST
test = pickle.load( open( "test.p", "rb" ), encoding='bytes')
#print(test.shape)

#lum_img = train[100,:,:,:]
#plt.imshow(lum_img)
#plt.show()

# Read the ID's and breeds in the csv file
labels_raw = pd.read_csv("dog-breed-identification/labels.csv", header=0, sep=',', quotechar='"')
labels_raw_y_train = pd.read_csv("dog-breed-identification/labels.csv", header=0, skiprows = 1, sep=',', quotechar='"')
#print(labels_raw_y_train)

# Filter out other breeds in the readin content except these breeds
labels_raw_sub = labels_raw[ (labels_raw["breed"] == 'doberman') | \
                            (labels_raw["breed"] == 'labrador_retriever') | \
                            (labels_raw["breed"] == 'yorkshire_terrier') | \
                            (labels_raw["breed"] == 'german_shepherd') | \
                            (labels_raw["breed"] == 'pug') ]


labels_freq_pd, count= np.unique(labels_raw_sub["breed"], return_counts=True)
count_sort_ind = np.argsort(-count)

main_labels = labels_freq_pd[count_sort_ind]
        
labels_raw_np = labels_raw["breed"].values
labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)
labels_filtered_index = np.where(labels_raw_np == main_labels)
labels_filtered = labels_raw.iloc[labels_filtered_index[0],:]

train_filtered = train[labels_filtered_index[0],:,:,:]
test_filtered = test[labels_filtered_index[0],:,:,:]


labels = labels_raw_sub["breed"].values
labels = labels.reshape(labels.shape[0],1)

labels_name, labels_bin = preprocess.matrix_Bin(labels = labels)
#print(labels_bin[0:10])

#for breed in range(len(labels_name)):
#    print('Breed {0} : {1}'.format(breed,labels_name[breed]))

    
if (not saved_model.is_file()):
    num_validation = 0.20
    X_train, X_validation, y_train, y_validation = train_test_split(train_filtered, labels_bin, test_size=num_validation, random_state=6)
    
    
    
    df_train_toUse, df_train_toPred, df_test_toUse, df_test_toPred = preprocess.train_test_creation(0.8, train_filtered, labels_bin)
    
    df_validation_toPred_cls = np.argmax(y_validation, axis=1)
    #print(df_validation_toPred_cls[0:9])
    
    # performing 'on the fly' data augmentation"
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,\
                             shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    
    history = model.fit_generator(
    	aug.flow(df_train_toUse, df_train_toPred, batch_size=32),
    	validation_data=(X_validation, y_validation),
        validation_steps=df_validation_toPred_cls // 32,
    	steps_per_epoch=len(df_train_toUse) // 32,
        shuffle=True,
    	epochs=50)
    
    
    # save model
    model.save(model_file)
else:
    model = keras.models.load_model(model_file)

input_image = sys.argv[1]
#input_image = "archive/images/Images/n02107142-Doberman/n02107142_1694.jpg"
#input_image = "archive/images/Images/n02107142-Doberman/n02107142_9621.jpg"
#input_image = "archive/images/Images/n02110958-pug/n02110958_1975.jpg"
#input_image = "archive/images/Images/n02106662-German_shepherd/n02106662_2810.jpg"
#input_image = "archive/images/Images/n02099712-Labrador_retriever/n02099712_619.jpg"
#input_image = "archive/images/Images/n02094433-Yorkshire_terrier/n02094433_478.jpg"

# load yolov3 model
ymodel = yolo.load_model('ymodel.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
photo_filename = input_image
# load and prepare image
image, image_w, image_h = yolo.load_image_pixels(photo_filename, (input_w, input_h))
# make prediction
yhat = ymodel.predict(image)
# summarize the shape of the list of arrays
print([a.shape for a in yhat])
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
# define the probability threshold for detected objects
class_threshold = 0.6
boxes = list()
for i in range(len(yhat)):
	# decode the output of the network
	boxes += yolo.decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
yolo.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
yolo.do_nms(boxes, 0.5)
# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# get the details of the detected objects
v_boxes, v_labels, v_scores = yolo.get_boxes(boxes, labels, class_threshold)
# summarize what we found
for i in range(len(v_boxes)):
	print("The object in the bounded box is a {} with {:.2f}% confidence \n\n".format(v_labels[i], v_scores[i]))
# draw what we found
yolo.draw_boxes(photo_filename, v_boxes, v_labels, v_scores)


if (len(v_boxes) == 1):
    if (dog_detector(input_image)):
        # new instance where we do not know the answer
        Xnew = np.asarray(load_image(input_image))
        # make a prediction
        ynew = model.predict(Xnew.reshape(Xnew.shape[0], image_resize, image_resize, 3))
        ynew_max = np.argmax(ynew)
        
        # predict age of the dog
        if (not age_saved_model.is_file()):
            cnn.test_age()
        else:
            amodel = keras.models.load_model(age_saved_model)
        
        img = load_img(input_image, target_size=(200, 200))
    	# convert to array
        img = img_to_array(img)
        dog_age = amodel.predict(img.reshape(1, 200, 200, 3))
        print(dog_age)
        # show the inputs and predicted outputs
        #print(ynew)
        #print(labels_name)
        print("\n")
        print("Predicted = {} with {:.2f}% confidence".format(labels_name[ynew_max], ynew[0, ynew_max] * 100))
    else:
        if (face_detector(input_image)):
            print("The object in the picture is a human.")
        else:
            print("The object in the picture is neither a dog nor a human.")
