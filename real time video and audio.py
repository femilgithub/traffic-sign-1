import cv2
from keras.models import load_model
import numpy as np
import pyttsx3

model = load_model("mymodel2.h5")

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)  # change the argument to 0 or -1 to use your camera
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

labels = open("labels.csv",'r').readlines()
labels = labels[1::]
lbl=[]
for label in labels:
    lbl.append(label.split(',')[1].rstrip('\n'))

engine = pyttsx3.init()

while True:
    success, imgOrignal = cap.read()
    if not success:
        break
    
    img = cv2.resize(imgOrignal, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)

    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    predictions = model.predict(np.expand_dims(img, axis=0))
    y_classes = [np.argmax(element) for element in predictions]
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOrignal, str(y_classes) + " " + str(lbl[y_classes[0]]), (120, 35), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
        
        # Speak the predicted label
        engine.say(lbl[y_classes[0]])
        engine.runAndWait()

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
