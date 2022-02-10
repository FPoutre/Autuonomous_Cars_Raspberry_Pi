from threading import Thread
import sys
import signal
import argparse

sys.path.append(r'/opt/ezblock')
from ezblock import __reset_mcu__
from picarmini import dir_servo_angle_calibration, set_dir_servo_angle
from picarmini import camera_servo1_angle_calibration, camera_servo2_angle_calibration
from picarmini import set_camera_servo1_angle, set_camera_servo2_angle
from picarmini import stop, backward

from lanefollower import LaneFollower
# from signdetector import SignDetector


def cleanup(sig, frame):
    print("Stopping all threads")
    laneFollower.kill = True
    # signDetector.kill = True
    laneFollower.join()
    # signDetector.join()
    print("All threads stopped")
    set_dir_servo_angle(0)
    stop()
    print("Goodbye !")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--legacy',
        action='store_false',
        help='Tells if LaneFollower should use legacy preprocessing or not.')
    args = parser.parse_args()

    __reset_mcu__()

    dir_servo_angle_calibration(0)
    camera_servo1_angle_calibration(-10)
    camera_servo2_angle_calibration(5)

    set_dir_servo_angle(0)
    set_camera_servo1_angle(0)
    set_camera_servo2_angle(0)
    backward(10)

    laneFollower = LaneFollower(-1, args.legacy)
    # signDetector = SignDetector(-1)

    laneFollower.start()
    # signDetector.start()

    signal.signal(signal.SIGINT, cleanup)
    signal.pause()
