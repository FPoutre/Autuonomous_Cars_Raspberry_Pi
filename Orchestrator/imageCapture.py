import sys
import time
import cv2
import numpy as np

sys.path.append(r'/opt/ezblock')
from picarmini import dir_servo_angle_calibration, set_dir_servo_angle
from picarmini import camera_servo1_angle_calibration, camera_servo2_angle_calibration
from picarmini import set_camera_servo1_angle, set_camera_servo2_angle


def frameCap():
    global cap
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def imgPreprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    # image = image / 255                # normalizing the pixel values 
    return image

if __name__ == "__main__":
    dir_servo_angle_calibration(0)
    camera_servo1_angle_calibration(-10)
    camera_servo2_angle_calibration(5)

    set_dir_servo_angle(0)
    set_camera_servo1_angle(0)
    set_camera_servo2_angle(0)

    cap = cv2.VideoCapture(0)

    oImg = frameCap()
    cv2.imwrite("../Data/capture.png", oImg)
    print("Saved capture to Data/capture.png")

    start_t = time.time_ns()
    pImg = imgPreprocess(oImg)
    pImg = cv2.resize(pImg, (320, 120))
    xImg = np.expand_dims(pImg, axis=0).astype('float32')
    total_t = time.time_ns() - start_t

    print("Preprocessing took {} ns".format(total_t)) 

    cv2.imwrite("../Data/preproc_capture.png", pImg)
    print("Saved preprocessed capture to Data/capture.png")
