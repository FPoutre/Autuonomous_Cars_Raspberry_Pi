from time import sleep
import sys
import threading
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import skimage

sys.path.append(r'/opt/ezblock')
from picarmini import backward, stop

class SignDetector(threading.Thread):

    def __init__(self, freq):
        threading.Thread.__init__(self)
        
        self.freq = freq
        self.kill = False

        self.speedLimit = 50
        backward(self.speedLimit)

        self.cap = cv2.VideoCapture(0)

        self.interpreter = tflite.Interpreter("../RecognitionModel/model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.input_shape = self.input_details[0]['shape']

    """
    Returns an int between 0 and 4 (included).
    0:'Speed limit (30km/h)'
    1:'Speed limit (50km/h)'
    2:'No overtaking'
    3:'Stop'
    4:'No entry'
    Only 0, 1 and 3 are needed here.
    """
    def predict(self):
        ret, img = self.cap.read()
        img = skimage.transform.resize(img, (30, 30))
        img = np.reshape(img, (-1,30,30,3)).astype('float32')

        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        self.interpreter.invoke()

        return np.argmax(self.interpreter.get_tensor(self.output_details[0]['index']))

    def run(self):
        while not self.kill:
            sleep(1/self.freq)
            prediction = self.predict()

            if prediction == 0:
                backward(30)
                self.speedLimit = 30
            elif prediction == 1:
                backward(50)
                self.speedLimit = 50
            elif prediction == 3:
                stop()
                sleep(3)
                backward(self.speedLimit)