import cv2
import csv
import random
import numpy as np

def imageReadLegacy(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def imgPreprocessLegacy(image):
    height, *_ = image.shape
    image = image[int(height/2):,:]  # remove top half of the image, as it is not relavant for lane following
    return image

def imageRead(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def imgPreprocess(image):
    height, *_ = image.shape
    image = image[int(height/2):,:]  # remove top half of the image, as it is not relavant for lane following
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image


if __name__=="__main__":
    database = []
    with open('../Data/DatabasePS4.csv', newline='') as csvfile:
        csv_dict_reader = csv.DictReader(csvfile)
        for row in csv_dict_reader:
            database.append(row)
    
    file = database[random.randint(1, len(database)-1)]

    oLFrame = imageReadLegacy("../Data/ImagesPS4/" + file["Images"] + ".jpg")
    cv2.imwrite("../Data/legacy.png", oLFrame)
    pLFrame = imgPreprocessLegacy(oLFrame)
    cv2.imwrite("../Data/preprocessed_legacy.png", pLFrame)

    oNFrame = imageRead("../Data/ImagesPS4/" + file["Images"] + ".jpg")
    cv2.imwrite("../Data/new.png", oNFrame)
    pNFrame = imgPreprocess(oNFrame)
    cv2.imwrite("../Data/preprocessed_new.png", pNFrame)