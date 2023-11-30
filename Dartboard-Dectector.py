import numpy as np
import cv2
import os
import sys
import argparse
import math

parser = argparse.ArgumentParser(description='DartBoard detection')
parser.add_argument('-viola_jones', '-v', action='store_true')
parser.add_argument('-name', '-n', type=str, default='')
args = parser.parse_args()

cascade_name = "Dartboardcascade/cascade.xml"

def convolution(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    padding = math.floor(kernel.shape[0] / 2)
    kh, kw, = kernel.shape
    h = image.shape[0]
    w = image.shape[1]
    outputX = w + 2 * padding
    outputY = h + 2 * padding
    output = np.zeros((outputY, outputX))
    paddedImage = np.zeros((outputY, outputX))
    paddedImage[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    for i in range(outputY - kh + 1):
        for f in range(outputX - kw + 1):
            v = np.sum(np.multiply(kernel, paddedImage[i:i+kh,f:f+kw]))
            output[i][f] = v

    return output[:h, :w]

def gradient_x(image):
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    return convolution(image, kernel)

def gradient_y(image):
    kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    return convolution(image, kernel)

def sobel_information(image):
    grad_x = gradient_x(image)
    grad_y = gradient_y(image)
    grad_magnitude = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))
    grad_direction = np.arctan2(grad_y, grad_x + 1e-10)
    return (grad_magnitude, grad_direction)

def thresholded_pixels(image, T):
    return np.argwhere(image > T)

def hough_circles(grad_direction, thresholded_grad_mag, shape, T):
    height, width = shape
    max_radius = 100
    H = np.zeros((height + max_radius, width + max_radius, max_radius))
    for pixel in thresholded_grad_mag:
        (y, x) = pixel
        for r in range(10, max_radius):
            x_comp = int(r * math.cos(grad_direction[y][x]))
            y_comp = int(r * math.sin(grad_direction[y][x]))
            H[y + y_comp][x + x_comp][r] += 1
            H[y - y_comp][x - x_comp][r] += 1  
    return np.argwhere(H > T)

def hough_ellipse(grad_direction, thresholded_grad_mag, shape, T):
    height, width = shape
    max_major_axis = max_minor_axis = 100
    H = np.zeros((height + max_major_axis, width + max_major_axis, max_major_axis, max_major_axis))
    for pixel in thresholded_grad_mag:
        (y, x) = pixel
        for r_a in range(10, max_major_axis):
            for r_b in range(10, r_a): # Semi-minor axis is always less than the semi-major axis
                for theta in range(0, 180):
                    alpha = grad_direction[y][x]
                    theta = theta * math.pi / 180.0
                    x_delta = int((r_a * math.cos(alpha) * math.cos(theta)) - (r_b * math.sin(alpha) * math.sin(theta)))
                    y_delta = int((r_a * math.cos(alpha) * math.sin(theta)) + (r_b * math.sin(alpha) * math.cos(theta)))
                    H[y + y_delta][x + x_delta][r_a][r_b] += 1
                    H[y - y_delta][x - x_delta][r_a][r_b] += 1  


def test_hough_circles(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    magnitude, direction = sobel_information(frame_gray)
    circles = hough_circles(direction, thresholded_pixels(magnitude, 200), frame_gray.shape, 10)
    for (y, x, r) in circles:
        image = cv2.circle(image, (x, y), r, (0, 0, 255), 1)
    cv2.imwrite("circles_detected.jpg", image)

def test_hough_ellipses(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    magnitude, direction = sobel_information(frame_gray)
    ellipses = hough_ellipse(direction, thresholded_pixels(magnitude, 200), frame_gray.shape, 10)
    for (x, y, r_a, r_b, angle) in ellipses:
        image = cv2.ellipse(image, (x, y), (r_a, r_b), angle, 0, 360, (0, 0, 255), 1)
    cv2.imwrite("ellipses_detected.jpg", image)

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
if (args.name != "" and args.viola_jones):
    ClassifySinglePhoto(imageName)
elif (args.name != ""):
    image = cv2.imread(imageName, 1)
    test_hough_ellipses(image)
    #test_hough_circles(image)
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