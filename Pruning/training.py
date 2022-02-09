# Imports

# python standard libraries
import os
import random

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
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.vgg16 import VGG16

# sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# imaging
import cv2
from imgaug import augmenters as img_aug


data = pd.read_csv("../Data/DatabasePS4.csv")
data = data[['Images','Steering','Speed']]
data.head()

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

imPath, sp, steer = loadData("../Data/ImagesPS4",data)
speeds = np.around(sp)
steer = np.around(steer)

# We check that all lists have the same size
print(len(imPath))
print(len(speeds))
print(len(steer))

# Now we shuffle the data to ameliorate the training
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

steerings = steerings_preprocess/35
steerings = steerings.astype('float32')
print(min(steerings))
print(max(steerings))

speeds = (speeds/25) - 1    # since speeds are in [0,50]
speeds = speeds.astype('float32')
print(min(speeds))
print(max(speeds))

speedsSteerings = np.array([speeds, steerings])
speedsSteerings = np.transpose(speedsSteerings)
print(speedsSteerings)


xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)

print('Total Training Images : ',len(xTrain))
print('Total Validation Images : ',len(xVal))
print('Total Images : ',len(imagesPath))

# This function serves to read the images in the correct way
def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

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

# put it together
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
    height, _ = image.shape
    image = image[int(height/2):,:]  # remove top half of the image, as it is not relavant for lane following
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = image / 255                # normalizing the pixel values 
    return image

def lane_following_model():
    model = Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 320, 1), padding='same'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.2),
      layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.2),
      layers.GlobalAvgPool2D(),
      layers.Dense(64, activation='relu'),
      # Output layer
      layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',loss='mse')
    
    return model


# vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(120,320, 1)) # include_top=False is to not keep the top layer
# for layer in vgg_model.layers:
#     layer.trainable = False
# x_vgg = Flatten()(vgg_model.output)
# d_vgg = Dense(50, activation='elu')(x_vgg)
# dense3_vgg = Dense(10, activation='elu')(d_vgg)
# output_vgg = Dense(1, activation='tanh')(dense3_vgg)
# vgg = Model(inputs=vgg_model.input, outputs=output_vgg)
# optimizer_vgg = Adam(learning_rate=1e-3) # lr is learning rate
# vgg.compile(loss='mse', optimizer=optimizer_vgg)

lane_following = lane_following_model()


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

ncol = 2
nrow = 2

X_train_batch, y_train_batch = next(image_data_generator(xTrain, yTrain, nrow, True))
X_valid_batch, y_valid_batch = next(image_data_generator(xVal, yVal, nrow, False))

batch_size=32

history = lane_following.fit(image_data_generator( xTrain, yTrain, batch_size=batch_size, is_training=True),
                              steps_per_epoch=len(xTrain)//8*batch_size,
                              epochs=16,
                              validation_data = image_data_generator( xVal, yVal, batch_size=batch_size, is_training=False),
                              validation_steps=30,
                              verbose = 1
                              )

# Save the model
model_json = lane_following.to_json()
with open("../LaneFollowingModel/source/model.json", "w") as json_file:
    json_file.write(model_json)
lane_following.save_weights("../LaneFollowingModel/source/model_weights.h5")
print("Saved model to disk")
