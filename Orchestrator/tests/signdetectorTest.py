from signdetector import SignDetector
from controller import PicarControl
import matplotlib.pyplot as plt

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

ret, img = signDetector.cap.read()
plt.title("Detected Sign : {}".format(predictedSpeed))
plt.axis('off')
plt.imshow(img)
plt.show()