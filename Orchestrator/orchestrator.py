from threading import Thread
import sys
import signal

sys.path.append(r'/opt/ezblock')
from picarmini import dir_servo_angle_calibration, set_dir_servo_angle
from picarmini import stop, backward

import lanefollower
import signdetector

def cleanup(sig, frame):
    laneFollowerThread.join()
    # signDetectorThread.join()
    stop()
    sys.exit(0)

if __name__ == "__main__":
    dir_servo_angle_calibration(0)
    set_dir_servo_angle(0)
    backward(10)

    laneFollower = lanefollower.LaneFollower(5)
    signDetector = signdetector.SignDetector(5)

    laneFollowerThread = Thread(target=lanefollower.continuousDetection, args=(laneFollower))
    # signDetectorThread = Thread(target=signdetector.continuousDetection, args=(signDetector))

    laneFollowerThread.start()
    # signDetectorThread.start()

    signal.signal(signal.SIGINT, cleanup)
    signal.pause()

    while True:
        continue