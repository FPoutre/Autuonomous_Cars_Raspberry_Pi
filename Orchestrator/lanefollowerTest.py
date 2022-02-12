import sys
import time

sys.path.append(r'/opt/ezblock')
from ezblock import __reset_mcu__
from picarmini import dir_servo_angle_calibration, set_dir_servo_angle

from lanefollower import LaneFollower

__reset_mcu__()
time.sleep(1) # Waiting for MCU to restart


laneFollower = LaneFollower()

# dir_servo_angle_calibration(3.35)
predictedAngle = laneFollower.predict()
set_dir_servo_angle(int(predictedAngle))
time.sleep(1)