# Imports

# python standard libraries
import os
import random
import argparse
import csv

# data processing
import numpy as np
import pandas as pd

# tensorflow
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# imaging
import cv2
from imgaug import augmenters as img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import model_from_json

# load json and create model
def load_model():
    json_file = open("../LaneFollowingModel/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../LaneFollowingModel/model_weights.h5")
    print("Model Loaded from disk")
    return loaded_model
    
    
# Convert the dataframe into lists. We created 3 differents lists:
# the images paths list, the speeds list and the steering angles list
def loadData(path, data):
  imagesPath = []
  speed = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append(path + "/" + indexed_data[0] + ".jpg")
    speed.append(float(indexed_data[2]))
    steering.append(float(indexed_data[1]))
  imagesPath = np.asarray(imagesPath)
  speed = np.asarray(speed)
  steering = np.asarray(steering)
  return imagesPath, speed, steering

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



def shuffle_images(imPath, steer):
    tmp = []

    for i in range(len(steer)):
      tmp.append([steer[i], imPath[i]])

    tmp = np.array(tmp)
    np.random.shuffle(tmp)

    steerings_preprocess = []
    imagesPath = []
    for i in range(len(steer)):
      steerings_preprocess.append((tmp[i][0]))
      imagesPath.append(tmp[i][1])

    steerings_preprocess = [float(k) for k in steerings_preprocess]
    steerings_preprocess = [int(k) for k in steerings_preprocess]
    steerings_preprocess = np.array(steerings_preprocess)

    imagesPath = np.array(imagesPath)
    
    print("images shuffled")
    return imagesPath, steerings_preprocess

def zoom(image):
    zoom = img_aug.Affine(scale=(1, 1.3))  # zoom from 100% (no zoom) to 130%
    image = zoom.augment_image(image)
    return image
    
def pan(image):
    # pan left / right / up / down about 10%
    pan = img_aug.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

def adjust_brightness(image):
    # increase or decrease brightness by 30%
    brightness = img_aug.Multiply((0.7, 1.3))
    image = brightness.augment_image(image)
    return image

def blur(image):
    kernel_size = random.randint(1, 5)  # kernel larger than 5 would make the image way too blurry
    image = cv2.blur(image,(kernel_size, kernel_size))
   
    return image

def random_flip(image, steering_angle):
    is_flip = random.randint(0, 1)
    #is_flip = 1      # uncomment if you want every image flipped for the training
    
    if is_flip == 1:
        # randomly flip horizontally
        image = cv2.flip(image,1)
        steering_angle = - steering_angle
   
    return image, steering_angle

def random_augment(image, steering_angle):
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = blur(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)
    image, steering_angle = random_flip(image, steering_angle)
    
    return image, steering_angle

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    image = image / 255                # normalizing the pixel values
    return image
   
def image_data_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering_angles = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]
            if is_training:
                # training: augment image
                image, steering_angle = random_augment(image, steering_angle)
              
            image = img_preprocess(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)
            
        yield(np.asarray(batch_images), np.asarray(batch_steering_angles))    
        
def define_model_for_pruning(lane_following_model, imagesPath, epochs, batch_size ):
    
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    validation_split = 0.2 # 20% of training set will be used for validation set.
    num_images = len(imagesPath) * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.80,
                                                   begin_step=0,
                                                   end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(lane_following_model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
                  
    return model_for_pruning


def pruned_model(model_for_pruning, xTrain, yTrain, xVal, yVal,epochs, batch_size):
    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep()
    ]

    model_for_pruning.fit(
        image_data_generator( xTrain, yTrain, batch_size=batch_size, is_training=True),
        steps_per_epoch=600,
        epochs=epochs,
        validation_data = image_data_generator( xVal, yVal, batch_size=batch_size, is_training=False),
        validation_steps=300,
        verbose = 1,
        callbacks=callbacks)
        
    return model_for_pruning


def convert_to_tflite_pruned_model(model_for_export):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    # Save the model

    with open('../LaneFollowingModel/pruned_model.tflite', 'wb') as f:
      f.write(pruned_tflite_model)
    
    print("TFlite Pruned Model Saved")


def convert_to_tflite_quantized_and_pruned_model(model_for_export):

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    with open('../LaneFollowingModel/dq_pruned_model.tflite', 'wb') as f:
      f.write(quantized_and_pruned_tflite_model)
      print("TFlite Dynamic Range Quantized and Pruned Model Saved")
     

def representative_dataset():
    global imagesPath
    index = random.randint(0, len(imagesPath))
    img = my_imread(imagesPath[index])
    img = img_preprocess(img)
    yield np.reshape(img,(-1,120,320,3)).astype('float32')


def convert_to_tflite_fiq_pruned_model(model):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_quant_model = converter.convert()

    with open('../LaneFollowingModel/fiq_pruned_model.tflite', 'wb') as f:
        f.write(tflite_quant_model)
        print("TFlite Full Integer Quantized and Pruned Model Saved")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        default='../Data/',
        help='directory containing images to be classified, as well as DatabasePS4.csv')
    parser.add_argument(
        '-b',
        '--batch_size',
        default=32,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '-e',
        '--epochs',
        default=2,
        help='Number of epoch')
    parser.add_argument(
        '--input_mean',
        default=127.5,
        type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5,
        type=float,
        help='input standard deviation')
    args = parser.parse_args()
    
    
    data = pd.read_csv(args.path + "DatabasePS4.csv")
    data = data[['Images','Steering','Speed']]
    
    # Transform string into float
    data['Steering'] = data['Steering'].astype(float)
    data['Speed'] = data['Speed'].astype(float)
    
    imPath, sp, steer = loadData(args.path + "ImagesPS4",data)
    speeds = np.around(sp)
    steer = np.around(steer)
    
    imagesPath, steerings_preprocess = shuffle_images(imPath, steer)
    
    steerings = steerings_preprocess/35
    steerings = steerings.astype('float32')
    
    xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)

    print('Total Training Images : ',len(xTrain))
    print('Total Validation Images : ',len(xVal))
    print('Total Images : ',len(imagesPath))
    
    lane_following_model = load_model()
    model_for_pruning = define_model_for_pruning(lane_following_model, imagesPath, args.epochs, args.batch_size)
    pruned_model = pruned_model(model_for_pruning, xTrain, yTrain, xVal, yVal,args.epochs, args.batch_size)
    
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    convert_to_tflite_pruned_model(model_for_export)
    convert_to_tflite_quantized_and_pruned_model(model_for_export)
    convert_to_tflite_fiq_pruned_model(model_for_export)
