from threading import Thread
import sys

import controller
import lanefollower

picar = controller.PicarControl()
laneFollower = lanefollower.LaneFollower(10, picar)

laneFollowerThread = Thread(target=lanefollower.continuousDetection, args=(laneFollower))

laneFollowerThread.start()

inputS = ""

while inputS != "exit":
    inputS = input("Enter \"exit\" to stop process.")

laneFollowerThread.join()
sys.exit()