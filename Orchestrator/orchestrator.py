from threading import Thread
import sys
import signal

sys.path.append(r'/opt/ezblock')
from picarmini import dir_servo_angle_calibration, set_dir_servo_angle
from picarmini import stop, backward

from obstacledetector import ObstacleDetector
from lanefollower import LaneFollower
# from signdetector import SignDetector


def cleanup(sig, frame):
    obstacleDetector.kill = True
    laneFollower.kill = True
    # signDetector.kill = True
    print("Stopping all threads")

    obstacleDetector.join()
    laneFollower.join()
    # signDetector.join()
    print("All threads stopped")

    set_dir_servo_angle(0)
    stop()
    print("Goodbye !")
    sys.exit(0)

if __name__ == "__main__":
    dir_servo_angle_calibration(0)
    set_dir_servo_angle(0)

    backward(10)

    obstacleDetector = ObstacleDetector()
    laneFollower = LaneFollower(-1)
    # signDetector = SignDetector(-1)

    obstacleDetector.start()
    laneFollower.start()
    # signDetector.start()

    signal.signal(signal.SIGINT, cleanup)
    signal.pause()