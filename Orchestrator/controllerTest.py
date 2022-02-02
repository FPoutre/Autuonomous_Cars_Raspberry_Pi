import sys
import time
sys.path.append(r'/opt/ezblock')
from picarmini import dir_servo_angle_calibration, set_dir_servo_angle
from picarmini import stop, backward, forward

if __name__ == "__main__":
    print("Calibration")
    dir_servo_angle_calibration(0)

    print("Forward")
    backward(10)

    print("Left 30° 1.5s")
    set_dir_servo_angle(-30)
    time.sleep(1.5)

    print("Reverse")
    forward(10)

    print("Right 30° 1.5s")
    set_dir_servo_angle(30)
    time.sleep(1.5)

    print("Stop")
    stop()
    set_dir_servo_angle(0)

    print("Done")