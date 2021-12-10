import picarmini

class PicarControl:

    def __init__(self):
        picarmini.dir_servo_angle_calibration(0)

    def setSpeed(self, speed):
        if speed == 0:
            picarmini.stop()
        elif speed > 0:
            picarmini.forward(speed)
        else:
            picarmini.backward(-speed)
    
    def turn(self, angle):
        picarmini.set_dir_servo_angle(angle)
