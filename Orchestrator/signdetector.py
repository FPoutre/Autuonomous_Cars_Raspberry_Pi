from time import sleep
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import skimage

class SignDetector:

    def __init__(self, freq, controller):
        self.freq = freq

        self.picar = controller
        self.speedLimit = 50

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

def continuousDetection(signDetector):
    while not False:
        sleep(1/signDetector.freq)
        prediction = signDetector.predict()

        if prediction == 0:
            signDetector.picar.setSpeed(30)
            signDetector.speedLimit = 30
        elif prediction == 1:
            signDetector.picar.setSpeed(50)
            signDetector.speedLimit = 50
        elif prediction == 3:
            signDetector.picar.setSpeed(0)
            sleep(3)
            signDetector.picar.setSpeed(signDetector.speedLimit)