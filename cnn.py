#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:24:19 2020

@author: nman
"""

from pathlib import Path
import matplotlib.pyplot as plt
import sys
from numpy import asarray
from PIL import Image

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator

# define cnn model for dog breed identification
def define_model():
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    
    input = Input(shape=(120,120,3),name = 'image_input')
    
    #Use the generated model 
    output_vgg16_conv = model(input)
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
	# add new classifier layers
    dropout1 = Dropout(0.2)(output_vgg16_conv)
    flat1 = Flatten()(dropout1)
    class1 = Dense(256, activation='relu', kernel_initializer='normal')(flat1)
    dropout2 = Dropout(0.5)(class1)
    output = Dense(5, activation='softmax')(dropout2)
	# define new model
    model = Model(inputs=input, outputs=output)
    model.summary()

	# compile model
    opt = SGD(lr=0.01)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# define cnn model for dog age prediction
def define_age_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='normal', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='normal', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='normal', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='normal'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.close()
    
# run the test harness for evaluating a model
def test_age():
	# define model
    model = define_age_model()
    model_file = "age_final_model.h5"
    
	# create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterator
    train_it = datagen.flow_from_directory('Expert_TrainEval',
		class_mode='binary', batch_size=32, target_size=(200, 200))
    test_it = datagen.flow_from_directory('test_age/test_dataset',
		class_mode='binary', batch_size=32, target_size=(200, 200))
	# fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), \
        validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=0)
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    # save model
    model.save(model_file)
    
'''
input_image = "archive/images/Images/n02107142-Doberman/n02107142_1306.jpg"
image = Image.open(input_image)
# convert image to numpy array
data = asarray(image)
#Xnew = np.asarray(load_image(input_image))
age_saved_model = Path("age_final_model.h5")
if (not age_saved_model.is_file()):
    test_age()
else:
    amodel = keras.models.load_model(age_saved_model)
dog_age = amodel.predict(data)
print(dog_age)
'''