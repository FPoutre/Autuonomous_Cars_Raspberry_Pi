import controller
from time import sleep


picar = controller.PicarControl()

picar.setSpeed(30)
sleep(3)
picar.setSpeed(0)
sleep(3)
picar.setSpeed(-30)
sleep(3)

picar.turn(-45)
sleep(3)
picar.setSpeed(30)
sleep(3)
picar.setSpeed(0)
sleep(3)
picar.setSpeed(-30)
sleep(3)

picar.turn(90)
sleep(3)
picar.setSpeed(30)
sleep(3)
picar.setSpeed(0)
sleep(3)
picar.setSpeed(-30)
sleep(3)

picar.turn(-45)
sleep(3)