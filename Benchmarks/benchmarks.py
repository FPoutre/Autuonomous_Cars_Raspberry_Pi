import argparse
import time
import csv
import random

import numpy as np
from math import floor
import statistics as stats
import cv2
#import tflite_runtime.interpreter as tflite

import tensorflow as tf
import matplotlib.pyplot as plt


def my_imread(image_path, useLegacy):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if useLegacy else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def img_preprocess(image, useLegacy):
    height, *_ = image.shape
    # remove top half of the image, as it is not relevant for lane following
    image = image[int(height/2):, :, :] if useLegacy else image[int(height/2):, :]
    if not useLegacy:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = image / 255 # normalizing the pixel values 
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--image_directory',
        default='../Data/',
        help='directory containing images to be classified, as well as DatabasePS4.csv')
    parser.add_argument(
        '-n',
        '--image_number',
        default=500,
        type=int,
        help='the number of images to benchmark on')
    parser.add_argument(
        '-m',
        '--model_file',
        default='../LaneFollowingModel/model.tflite',
        help='.tflite model to be executed')
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
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Tells if LaneFollower should use legacy preprocessing or not.')
    args = parser.parse_args()

    database = []
    with open(args.image_directory + '/DatabasePS4.csv', newline='') as csvfile:
        csv_dict_reader = csv.DictReader(csvfile)
        for row in csv_dict_reader:
            database.append(row)

    interpreter = tf.lite.Interpreter(model_path=args.model_file, num_threads=4) # Cortex A72 has 4 logical cores.
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    input_shape = input_details[0]['shape']

    total_start_time = time.time()
    time_list = []
    offset_list = []

    for i in range(args.image_number):
        minutes = floor(time.time() - total_start_time)//60
        seconds = floor(time.time() - total_start_time)%60
        print("Image {}/{}, {}m{}s".format(i+1, args.image_number, minutes, seconds), end='\r')
        file = database[random.randint(1, len(database)-1)]
        img = my_imread("{}ImagesPS4/{}.jpg".format(args.image_directory, file["Images"]), args.legacy)
        img = img_preprocess(img, args.legacy)
        img = np.reshape(img,(-1,120,320,3)).astype('float32') if args.legacy else np.reshape(img,(-1,120,320,1)).astype('float32')
        interpreter.set_tensor(input_details[0]['index'], img)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        predicted_angle = 35*interpreter.get_tensor(output_details[0]['index'])[0][0]

        time_list.append(stop_time - start_time)
        offset_list.append(np.abs(predicted_angle - float(file["Steering"])))
    
    print("Processing image: {}/{}, {}m{}s".format(args.image_number, args.image_number, minutes, seconds))
    print("Average time (per prediction) : {} ms".format(1000*stats.mean(time_list)))
    print("Average prediction offset : {}°".format(stats.mean(offset_list)))
