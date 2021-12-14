from time import sleep
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

class LaneFollower:

    def __init__(self, freq, controller):
        self.freq = freq

        self.picar = controller

        self.cap = cv2.VideoCapture(0)

        self.interpreter = tflite.Interpreter("../LaneFollowingModel/model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self):
        ret, img = self.cap.read()
        img = imgPreprocess(img)
        img = np.reshape(img,(-1,120,320,3)).astype('float32')
        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        self.interpreter.invoke()

        return 35*self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

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
