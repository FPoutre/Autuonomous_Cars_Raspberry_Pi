import sys
import threading

sys.path.append(r'/opt/ezblock')
from picarx import stop, backward
from ezblock import Pin, Ultrasonic

class ObstacleDetector(threading.Thread):

    STOP_THRESH = 15

    def __init__(self):
        threading.Thread.__init__(self)

        self.kill = False
        self.speed = 10
        self.stopped = False

        d0 = Pin("D0")
        d1 = Pin("D1")

        self.detector = Ultrasonic(d0, d1)
    
    def run(self):
        while not self.kill:
            if not self.stopped:
                if self.detector.read() < ObstacleDetector.STOP_THRESH:
                    stop()
                    self.stopped = True
            else:
                if self.detector.read() > ObstacleDetector.STOP_THRESH:
                    backward(self.speed)
                    self.stopped = False