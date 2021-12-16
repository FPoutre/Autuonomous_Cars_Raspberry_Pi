from signdetector import SignDetector
from controller import PicarControl

picar = PicarControl()
signDetector = SignDetector(1, picar)

sign_classes = {
    0:'Speed limit (30km/h)', 
    1:'Speed limit (50km/h)',  
    2:'No Overtaking', 
    3:'Stop', 
    4:'No entry' 
}

predictedSpeed = sign_classes[signDetector.predict()]
print("Detected Sign : {}".format(predictedSpeed))