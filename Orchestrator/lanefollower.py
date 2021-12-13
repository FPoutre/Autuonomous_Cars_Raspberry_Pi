from time import sleep
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
from controller import PicarControl

class LaneFollower:

    def __init__(self, freq, controller):
        self.freq = freq

        self.picar = controller

        self.interpreter = tflite.Interpreter("../LaneFollowingModel/model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = self.input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.cap = cv2.VideoCapture(0)

    def predict(self):
        ret, img = self.cap.read()
        img = imgPreprocess(img)
        np.reshape(img, (self.width, self.height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return 35*output_data[0][0]

def imgPreprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    image = image / 255                # normalizing the pixel values 
    return image

def continuousDetection(laneFollower):
    while not False:
        sleep(1/laneFollower.freq)
        angle = laneFollower.predict()
        laneFollower.picar.turn(angle)
