import sys
import time
import signal

sys.path.append(r'/opt/ezblock')
from picarx import set_dir_servo_angle
from picarx import stop, backward, forward

from lanefollower import LaneFollower

def cleanup(sig, frame):
    print("\nGoodbye !")
    sys.exit(0)

if __name__=="__main__":
    laneFollower = LaneFollower()

    signal.signal(signal.SIGINT, cleanup)

    while True:
        predictedAngle = laneFollower.predict()
        set_dir_servo_angle(predictedAngle)