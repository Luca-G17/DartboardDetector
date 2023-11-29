import numpy as np
import cv2
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='DartBoard detection')
parser.add_argument('-name', '-n', type=str, default='Dartboard/dart0.jpg')
args = parser.parse_args()

cascade_name = "Dartboardcascade/cascade.xml"

def IoUScore(b0, b1):
    # box = (x, y, w, h)
    # Area of Overlap / Area of Union
    minx = min(b0[0], b1[0])
    maxx = max(b0[0] + b0[2], b1[0] + b1[2])
    miny = min(b0[1], b1[1])
    maxy = max(b0[1] + b0[3], b1[1] + b1[3])

    overlap = (maxx - minx) * (maxy - miny) 



def detectAndDisplay(frame, truths, imageN):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    dartboards = model.detectMultiScale(frame_gray, scaleFactor=1.10, minNeighbors=8, flags=0, minSize=(30,30), maxSize=(300,300))
    print(len(dartboards))

    for i in range(0, len(dartboards)):
        start_point = (dartboards[i][0], dartboards[i][1])
        end_point = (dartboards[i][0] + dartboards[i][2], dartboards[i][1] + dartboards[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    for i in range(len(truths[imageN])):
        start_point = (truths[imageN][i][0], truths[imageN][i][1])
        end_point = (truths[imageN][i][0] + truths[imageN][i][2], truths[imageN][i][1] + truths[imageN][i][3])
        colour = (255, 0, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)


def readGroundTruths(filename):
    boxes = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            box = line.split(',')
            boxTuple = (int(box[1]), int(box[2]), int(box[3]), int(box[4]))
            if len(boxes) <= int(box[0]):
                boxes.append([boxTuple])
            else:
                boxes[int(box[0])].append(boxTuple)
    return boxes

imageName = args.name

if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

frame = cv2.imread(imageName, 1)

if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)


model = cv2.CascadeClassifier(cascade_name)
ground_truths = readGroundTruths("Dartboard/groundtruths.txt")
imageNo = int((imageName.split('dart')[1]).split('.jpg')[0])

detectAndDisplay(frame, ground_truths, imageNo)

cv2.imwrite("detected.jpg", frame)

# TP = True Positive, FP = False Positives, FN = False Negatives
# Use a thresholded IoU for classification
# TP, FP, FN
# F_1 = 2TP / (2TP + FP + FN)
#
# For each 'detected' dartboard det
# {
# found = false
# For each ground truth dartboard (dart, located)
# {
# if (IoU(det, dart) > T):
#   TP++
#   remove det from list
#   located = true
#   found = true
# }
# if !found:
#   FP++
# }
# for each ground truth dartboard (dart, located)
# if (!located)
#   FN++