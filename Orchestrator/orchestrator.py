from threading import Thread
import sys

import controller
import lanefollower
import signdetector

picar = controller.PicarControl()
laneFollower = lanefollower.LaneFollower(5, picar)
signDetector = signdetector.SignDetector(5, picar)

laneFollowerThread = Thread(target=lanefollower.continuousDetection, args=(laneFollower))
signDetectorThread = Thread(target=signdetector.continuousDetection, args=(signDetector))

laneFollowerThread.start()
signDetectorThread.start()

inputS = ""

while inputS != "exit":
    inputS = input("Enter \"exit\" to stop process.")

laneFollowerThread.join()
signDetectorThread.join()
sys.exit()