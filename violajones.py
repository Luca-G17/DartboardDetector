import cv2
import argparse
import sys
import os
import numpy as np

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    dartboards = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=2, flags=1)#, scaleFactor=1.6, minNeighbors=2, flags=4, minSize=(40,40), maxSize=(300,300))
    print(len(dartboards))
    for i in range(0, len(dartboards)):
        start_point = (dartboards[i][0], dartboards[i][1])
        end_point = (dartboards[i][0] + dartboards[i][2], dartboards[i][1] + dartboards[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

cascade_name = "Dartboardcascade/cascade.xml"
parser = argparse.ArgumentParser(description='Dartboard Detection')
parser.add_argument('-name', '-n', type=str, default='Dartboard/dart0.jpg')
args = parser.parse_args()

imageName = args.name

if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

frame = cv2.imread(imageName, 1)

if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)

model = cv2.CascadeClassifier(cascade_name)

detectAndDisplay(frame)
cv2.imwrite("detected.jpg", frame)
