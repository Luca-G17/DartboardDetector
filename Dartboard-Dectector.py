import numpy as np
import cv2
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='DartBoard detection')
parser.add_argument('-name', '-n', type=str, default='')
args = parser.parse_args()

cascade_name = "Dartboardcascade/cascade.xml"

def IoUScore(b0, b1):
    # box = (x, y, w, h)
    # Area of Overlap / Area of Union
    # Union = Area(b0) + Area(b1) - Overlap
    maxx = max(b0[0], b1[0])
    minx = min(b0[0] + b0[2], b1[0] + b1[2])
    maxy = max(b0[1], b1[1])
    miny = min(b0[1] + b0[3], b1[1] + b1[3])
    x_dist = minx - maxx
    y_dist = miny - maxy
    if x_dist <= 0 or y_dist <= 0:
        return 0
    
    overlap = x_dist * y_dist
    union = (b0[2] * b0[3]) + (b1[2] * b1[3]) - overlap
    return overlap / union

def ScorePredictions(predictions, groundtruths):
    scores = { "TP": 0, "FP": 0, "FN": 0, "F1": 0}
    truthLocated = np.full(len(groundtruths), False)
    threshold = 0.2
    for p in predictions:
        found = False
        for i in range(len(groundtruths)):
            truth = groundtruths[i]
            if (IoUScore(p, truth) > threshold):
                scores['TP'] += 1
                found = True
                truthLocated[i] = True

        if found == False:
            scores['FP'] += 1
    for l in truthLocated:
        if l == False:
            scores['FN'] += 1

    scores['F1'] = 2 * scores['TP'] / ((2 * scores['TP']) + scores['FP'] + scores['FN'])
    return scores

def PrettyPrintScores(scores):
    print()
    print(f"Image   | True Positive Rate | F1 Score")
    print("--------------------------------------------")
    averages = [0, 0]
    for i in range(len(scores)):
        score = scores[i]
        TPR = float(score["TP"]) / float(score["TP"] + score["FN"])
        F1 = score["F1"]
        averages[0] += TPR
        averages[1] += F1
        print("{:<8}| {:<19}| {:.3f}".format(i, TPR, F1))
    print("--------------------------------------------")
    print("Average | {:<19}| {:.3f}".format(averages[0] / len(scores), averages[1] / len(scores)))

def PrettyPrintScore(scores, imageN):
    fill = '-'
    align = '<'
    width = 10

    print()
    print(f"Image {imageN} - Performance Scores:")
    print("Metric          | Score")
    print(f"-----------------{fill:{fill}{align}{width}}")
    print("True Positives  | {:<10}".format(scores["TP"]))
    print("False Positives | {:<10}".format(scores["FP"]))
    print("False Negatives | {:<10}".format(scores["FN"]))
    print("F1 Score        | {:.3f}".format(scores["F1"]))

def detectAndDisplay(model, frame, truths, imageN):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    dartboards = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=8, flags=0, minSize=(30,30), maxSize=(300,300))
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
        colour = (0, 0, 255)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    scores = ScorePredictions(dartboards, truths[imageN])
    PrettyPrintScore(scores, imageN)
    return scores

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

def ClassifyMultiplePhotos(name_range, folder='Dartboard/', prefix='dart', ext='jpg'):
    model = cv2.CascadeClassifier(cascade_name)
    ground_truths = readGroundTruths("Dartboard/groundtruths.txt")
    scores = []
    (start, stop) = name_range
    for i in range(start, stop + 1):
        filename = f'{folder}{prefix}{i}.{ext}'
        if (not os.path.isfile(filename)):
            print('No such file')
            sys.exit(1)
        frame = cv2.imread(filename, 1)
        scores.append(detectAndDisplay(model, frame, ground_truths, i))
        cv2.imwrite(f"Detected/{prefix}{i}.{ext}", frame)
    PrettyPrintScores(scores)
        
def ClassifySinglePhoto(filepath):
    if (not os.path.isfile(filepath)) or (not os.path.isfile(cascade_name)):
        print('No such file')
        sys.exit(1)

    frame = cv2.imread(filepath, 1)

    if not (type(frame) is np.ndarray):
        print('Not image data')
        sys.exit(1)

    split = filepath.split('/')
    filename = split[len(split) - 1]
    model = cv2.CascadeClassifier(cascade_name)
    ground_truths = readGroundTruths("Dartboard/groundtruths.txt")
    imageNo = int((filepath.split('dart')[1]).split('.jpg')[0])
    detectAndDisplay(model, frame, ground_truths, imageNo)
    cv2.imwrite(f"Detected/{filename}", frame)

imageName = args.name
if (args.name != ""):
    ClassifySinglePhoto(imageName)
else:
    ClassifyMultiplePhotos((0, 15))
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