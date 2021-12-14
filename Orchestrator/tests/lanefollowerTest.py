from lanefollower import LaneFollower
from controller import PicarControl
import matplotlib.pyplot as plt

picar = PicarControl()
laneFollower = LaneFollower(1, picar)

predictedAngle = laneFollower.predict()

ret, img = laneFollower.cap.read()
plt.title("predicted angle : {}°".format(predictedAngle))
plt.axis('off')
plt.imshow(img)
plt.show()