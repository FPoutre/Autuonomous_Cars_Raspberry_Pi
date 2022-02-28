from threading import Thread
import sys
import signal
import argparse
import time

sys.path.append(r'/opt/ezblock')
from picarx import camera_servo1_angle_calibration, camera_servo2_angle_calibration
from picarx import set_camera_servo1_angle, set_camera_servo2_angle
from picarx import dir_servo_angle_calibration, set_dir_servo_angle
from picarx import stop, backward, forward

from lanefollower import LaneFollower
# from signdetector import SignDetector
from obstacledetector import ObstacleDetector


def cleanup(sig, frame):
    global args
    print("Stopping all threads")

    laneFollower.kill = True
    if not args.demo :
        # signDetector.kill = True
        obstacleDetector.kill = True

    laneFollower.join()
    if not args.demo :
        # signDetector.join()
        obstacleDetector.join()
    
    print("All threads stopped")
    set_dir_servo_angle(0)
    stop()
    
    print("Goodbye !")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Tells if LaneFollower should use legacy preprocessing or not.')
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Tells if vehicle should move or not for demonstration purposes.')
    args = parser.parse_args()

    dir_servo_angle_calibration(3.35)
    camera_servo1_angle_calibration(-6)
    camera_servo2_angle_calibration(5)

    set_dir_servo_angle(0)
    set_camera_servo1_angle(0)
    set_camera_servo2_angle(0)
    if not args.demo :
        backward(10)

    laneFollower = LaneFollower(delay=0 ,useLegacy=args.legacy)
    if not args.demo :
        # signDetector = SignDetector()
        obstacleDetector = ObstacleDetector()

    laneFollower.start()
    if not args.demo :
        # signDetector.start()
        obstacleDetector.start()

    signal.signal(signal.SIGINT, cleanup)
    signal.pause()