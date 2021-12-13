import argparse
import time
import csv
import random

import numpy as np
from math import floor
import statistics as stats
from PIL import Image
import tflite_runtime.interpreter as tflite


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
        '--num_threads', 
        default=4, 
        type=int, 
        help='number of threads')
    args = parser.parse_args()

    database = []
    with open(args.image_directory + '/DatabasePS4.csv', newline='') as csvfile:
        csv_dict_reader = csv.DictReader(csvfile)
        for row in csv_dict_reader:
            database.append(row)

    interpreter = tflite.Interpreter(
        model_path=args.model_file, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    total_start_time = time.time()
    time_list = []
    offset_list = []

    for i in range(args.image_number):
        minutes = floor(time.time() - total_start_time)//60
        seconds = floor(time.time() - total_start_time)%60
        print("Image {}/{}, {}m{}s".format(i+1, args.image_number, minutes, seconds), end='\r')
        file = database[random.randint(1, len(database)-1)]
        img = Image.open("{}ImagesPS4/{}.jpg".format(args.image_directory, file["Images"])).resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - args.input_mean) / args.input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        predicted_angle = 35*interpreter.get_tensor(output_details[0]['index'])[0][0]

        time_list.append(stop_time - start_time)
        offset_list.append(np.abs(predicted_angle - float(file["Steering"])))
    
    print("Processing image: {}/{}, {}m{}s".format(args.image_number, args.image_number, minutes, seconds))
    print("Average time (per prediction) : {} µs".format(1000*stats.mean(time_list)))
    print("Average prediction offset : {}°".format(stats.mean(offset_list)))