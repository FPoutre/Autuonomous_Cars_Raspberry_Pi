import time
import sys
import threading
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

sys.path.append(r'/opt/ezblock')
from picarx import dir_servo_angle_calibration, set_dir_servo_angle
from picarx import stop, backward, forward

class LaneFollower(threading.Thread):

    def __init__(self, delay=-1, useLegacy=False):
        threading.Thread.__init__(self)

        self.delay = delay
        self.kill = False
        self.useLegacy = useLegacy

        self.cap = cv2.VideoCapture(0)

        # Cortex A72 has 4 logical cores.
        self.interpreter = tflite.Interpreter("../LaneFollowingModel/dq/model_old.tflite", num_threads=4) if self.useLegacy else tflite.Interpreter("../LaneFollowingModel/dq/model.tflite", num_threads=4)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def frameCap(self):
        ret, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.useLegacy else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def imgPreprocess(self, image):
        height, *_ = image.shape
        # remove top half of the image, as it is not relevant for lane following
        image = image[int(height/2):, :, :] if self.useLegacy else image[int(height/2):, :]
        if not self.useLegacy:
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image = image / 255 # normalizing the pixel values 
        return image

    def predict(self):
        img = self.frameCap()
        img = self.imgPreprocess(img)
        img = cv2.resize(img, (320, 120))

        img = np.expand_dims(img, axis=0).astype('float32')
        if not self.useLegacy:
            img = img[..., np.newaxis]

        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        start_t = time.time_ns()
        self.interpreter.invoke()
        duration = time.time_ns() - start_t

        res = 35*self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        print("Predicted Angle: {}° ({}ms)".format(res, duration/1000000))

        return res

    def run(self):
        while not self.kill:
            predictedAngle = self.predict()
            set_dir_servo_angle(predictedAngle)

            if self.delay > 0:
                time.sleep(self.delay)