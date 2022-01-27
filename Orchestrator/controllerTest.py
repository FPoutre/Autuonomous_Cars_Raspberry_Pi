from controller import PicarControl
from time import sleep


picar = PicarControl()

def backAndForth():
    picar.setSpeed(30)
    sleep(3)
    picar.setSpeed(0)
    sleep(3)
    picar.setSpeed(-30)
    sleep(3)
    picar.setSpeed(0)

def leftTurn():
    picar.turn(-45)
    sleep(1)
    backAndForth()

def rightTurn():
    picar.turn(90)
    sleep(1)
    backAndForth()

backAndForth()
leftTurn()
rightTurn()
picar.turn(-45)
sleep(3)
