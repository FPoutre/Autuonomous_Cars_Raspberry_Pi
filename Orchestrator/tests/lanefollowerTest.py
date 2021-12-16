from lanefollower import LaneFollower
from controller import PicarControl

picar = PicarControl()
laneFollower = LaneFollower(1, picar)

predictedAngle = laneFollower.predict()
print("Predicted Angle : {}°".format(predictedAngle))