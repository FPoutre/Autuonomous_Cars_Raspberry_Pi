from time import sleep
import sys
import threading
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

sys.path.append(r'/opt/ezblock')
from picarmini import set_dir_servo_angle

class LaneFollower(threading.Thread):

    def __init__(self, freq):
        threading.Thread.__init__(self)

        self.freq = freq
        self.kill = False

        self.cap = cv2.VideoCapture(0)

        self.interpreter = tflite.Interpreter("../LaneFollowingModel/model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def frameCap(self):
        ret, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def imgPreprocess(self, image):
        height, _, _ = image.shape
        image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
        image = image / 255                # normalizing the pixel values 
        return image

    def predict(self):
        img = self.frameCap()
        img = self.imgPreprocess(img)
        img = cv2.resize(img, (320, 120))
        img = np.expand_dims(img, axis=0).astype('float32')
        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        self.interpreter.invoke()

        res = 35*self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        print("Predicted Angle: {}°".format(res))

        return res

    def run(self):
        while not self.kill:
            if self.freq != -1:
                sleep(1/self.freq)
            angle = self.predict()
            set_dir_servo_angle(angle)
