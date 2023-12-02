import numpy as np
import cv2
import os
import sys
import argparse
import math
import random

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
    cv2.imwrite("grad_magnitude.jpg", (grad_magnitude > 200) * 255)
    grad_direction = np.arctan2(grad_y, grad_x + 1e-10)
    return (grad_magnitude, grad_direction)

def thresholded_pixels(image, T):
    return np.argwhere(image > T)

def hough_circles(grad_direction, thresholded_grad_mag, shape, T):
    height, width = shape
    max_radius = 150
    H = np.zeros((height + max_radius, width + max_radius, max_radius))
    for pixel in thresholded_grad_mag:
        (y, x) = pixel
        for r in range(30, max_radius):
            x_comp = int(r * math.cos(grad_direction[y][x]))
            y_comp = int(r * math.sin(grad_direction[y][x]))
            H[y + y_comp][x + x_comp][r] += 1
            H[y - y_comp][x - x_comp][r] += 1
    circles = np.argwhere(H > T)
    
    all_circles = np.argwhere(H > 0)
    hough_space_image = np.zeros((height + max_radius, width + max_radius))
    for circle in all_circles:
        (y, x, r) = circle
        hough_space_image[y][x] += 2
    cv2.imwrite("circle_H_space.jpg", hough_space_image)

    averages = []
    cluster_radius = 50
    while(len(circles) > 0):
        average = list(circles[0])
        [ay, ax, ar] = average
        average.append(H[ay][ax][ar])
        removed = [0]
        for i in range(1, len(circles)):
            (ay, ax, ar, an) = average
            (y, x, r) = circles[i]
            n = H[y][x][r]
            acoords = np.array([ay, ax])
            coords = np.array([y, x])
            d2 = np.dot(acoords - coords, acoords - coords)
            if (d2 < cluster_radius ** 2):
                new_coords = ((acoords * an) + (coords * n)) / (an + n)
                new_radius = ((ar * an) + (r * n)) / (an + n)
                average = [int(new_coords[0]), int(new_coords[1]), int(new_radius), an + n]
                removed.append(i)

        circles = np.delete(circles, removed, axis=0)
        averages.append(average)
    return averages


def EllipseSimiliarity(c0, r0, c1, r1):
    rDiff = (float(r0[0]) / r0[1]) - (float(r1[0]) / r1[1])
    cDiff2 = np.dot(c0 - c1, c0 - c1)

    # 10 Pixel Radius for centre
    # 0.1 Difference in radii   
    return cDiff2 <= 10 and rDiff <= 0.1

def hough_ellipse(grad_direction, thresholded_grad_mag, shape, T, radii_range):
    height, width = shape
    min_radius, max_radius = radii_range
    H = np.zeros((height + max_radius, width + max_radius, max_radius - min_radius, max_radius - min_radius))
    hough_space_image = np.zeros((height + max_radius, width + max_radius))
    for pixel in thresholded_grad_mag:
        (y, x) = pixel
        for r_a in range(0, max_radius - min_radius):
            for r_b in range(0, max_radius - min_radius): # Semi-minor axis is always less than the semi-major axis
                alpha = grad_direction[y][x]
                x_delta = int((r_a + min_radius) * math.cos(alpha))
                y_delta = int((r_b + min_radius) * math.sin(alpha))
                H[y + y_delta][x + x_delta][r_a][r_b] += 1
                H[y - y_delta][x - x_delta][r_a][r_b] += 1
                hough_space_image[y + y_delta][x + x_delta] += 0.1
                hough_space_image[y - y_delta][x - x_delta] += 0.1
   
    es = np.argwhere(H > T)
    cv2.imwrite("ellipse_H_space.jpg", hough_space_image)

    remaining = np.copy(es)
    averages = []

    cluster_radius = 70
    print(remaining)
    while(len(remaining) > 0):
        average = list(remaining[0])
        [ay, ax, ar_a, ar_b] = average
        average.append(H[ay][ax][ar_a][ar_b])
        removed = [0]
        for i in range(1, len(remaining)):
            (ay, ax, ar_a, ar_b, an) = average
            (y, x, r_a, r_b) = remaining[i]
            n = H[y][x][r_a][r_b]
            acoords = np.array([ay, ax])
            aradii = np.array([ar_a, ar_b])
            coords = np.array([y, x])
            radii = np.array([r_a, r_b])

            d2 = np.dot(acoords - coords, acoords - coords)
            if (d2 < (cluster_radius ** 2)):
                new_coords = ((acoords * an) + (coords * n)) / (an + n)
                new_radii = ((aradii * an) + (radii * n)) / (an + n)
                average = [int(new_coords[0]), int(new_coords[1]), int(new_radii[0]), int(new_radii[1]), an + n] 
                removed.append(i)
        remaining = np.delete(remaining, removed, axis=0)
        averages.append(average)
    return averages

def AngleBetweenPoints(v0, v1, v2):
    return (180 / math.pi) * np.arccos(np.dot(v1 - v0, v1 - v2) / (np.linalg.norm(v1 - v0) * np.linalg.norm(v1 - v2)))

def TriangleHas18DegreeAngle(triangle, tolerance=10):
    angles = []
    angles.append(AngleBetweenPoints(triangle[0], triangle[1], triangle[2]))
    angles.append(AngleBetweenPoints(triangle[1], triangle[2], triangle[0]))
    angles.append(AngleBetweenPoints(triangle[2], triangle[0], triangle[1]))
    for a in angles:
        if (a < 18 + tolerance or a > 18 - tolerance):
            return True
    return False

def DetectDartboardsFromTriangles(frame):
    triangles = ContourPolygons(frame, 150)
    
    # Dartboards have 20 triangular segments, 360 / 20 = 18
    # Get a list of triangles that have an angle of around 18 degrees
    triangles = [np.array(t) for t in triangles if TriangleHas18DegreeAngle(np.array(t))]
    centres = np.array([(t[0] + t[1] + t[2]) / 3 for t in triangles])

    cluster_radius = 30
    clusters = []
    while (len(centres) > 0):
        average = list(centres[0])
        average.append(0)
        removed = [0]
        for i in range(1, len(centres)):
            centre = np.array(centres[i])
            av = np.array(average[:2])
            d2 = np.dot(centre - av, centre - av)
            if (d2 < cluster_radius ** 2):
                new_average = (av * average[2] + centre) / (average[2] + 1)
                average = [new_average[0], new_average[1], average[2] + 1]
                removed.append(i)
        centres = np.delete(centres, removed, axis=0)
        clusters.append(average)
    
    clusters = [c for c in clusters if c[2] >= 3]
    print(len(clusters))
    return [[int(c[0] - cluster_radius), int(c[1] - cluster_radius), 2 * cluster_radius, 2 * cluster_radius] for c in clusters]

def ContourPolygons(frame, T):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(frame_gray, T, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tris = []
    contour_output = np.copy(frame)
    for contour in contours:
        if (cv2.contourArea(contour) > 10):
            poly = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
            if (len(poly) == 3):
                cv2.drawContours(contour_output, [poly], 0, (0, 0, 255), 2)
                tris.append([poly[0][0], poly[1][0], poly[2][0]])
    cv2.imwrite("countour_test.jpg", contour_output)
    return tris

# TODO: Generalise this for ellipses
def LargeClusterCircles(circles, cluster_radius):
    clusters = []
    while(len(circles) > 0):
        average = list(circles[0])
        average.append(0)
        removed = [0]
        for i in range(1, len(circles)):
            average_coords = np.array(average[:2])
            coords = circles[i][:2]
            d2 = np.dot(average_coords - coords, average_coords - coords)
            if (d2 < cluster_radius ** 2):
                new_coords = ((average_coords * average[3]) + coords) / (average[3] + 1)
                new_radius = ((average[2] * average[3]) + circles[i][2]) / (average[3] + 1)
                average = [int(new_coords[0]), int(new_coords[1]), int(new_radius), average[3] + 1]
                removed.append(i)
        circles = np.delete(circles, removed, axis=0)
        clusters.append(average[:3])
        print(average[3])
    print(clusters)
    return clusters

def test_hough_circles(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    magnitude, direction = sobel_information(frame_gray)
    circles = hough_circles(direction, thresholded_pixels(magnitude, 200), frame_gray.shape, 6)
    for (y, x, r) in circles:
        image = cv2.circle(image, (x, y), r, (0, 0, 255), 1)
    cv2.imwrite("circles_detected.jpg", image)

def test_hough_ellipses(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    magnitude, direction = sobel_information(frame_gray)
    #ellipses = better_hough_ellipse(direction, thresholded_pixels(magnitude, 250), frame_gray.shape, 10)
    # ellipses = randomized_hough_ellipse(direction, thresholded_pixels(magnitude, 200))
    min_radius = 30
    ellipses = hough_ellipse(direction, thresholded_pixels(magnitude, 250), frame_gray.shape, 11, (min_radius, 100))
    height, width = frame_gray.shape
    centres = np.zeros((height + min_radius, width + min_radius))
    for (y, x, r_a, r_b) in ellipses:
        centres[y][x] += 30
    cv2.imwrite("ellipse_centres.jpg", centres)
            
    for (y, x, r_a, r_b) in ellipses:
        print(f"Ellipse: ({x}, {y}), ({r_a}, {r_b})")
        image = cv2.ellipse(image, (int(x), int(y)), (int(r_a + min_radius), int(r_b + min_radius)), 0, 0, 360, (0, 0, 255), 1)
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
    threshold = 0.1
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


def DrawGroundTruth(frame, truths, imageN):
    for i in range(len(truths[imageN])):
        start_point = (truths[imageN][i][0], truths[imageN][i][1])
        end_point = (truths[imageN][i][0] + truths[imageN][i][2], truths[imageN][i][1] + truths[imageN][i][3])
        colour = (0, 0, 255)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

def ViolaJonesDetector(model, frame, truths, imageN):
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

    DrawGroundTruth(frame, truths, imageN)
    scores = ScorePredictions(dartboards, truths[imageN])
    PrettyPrintScore(scores, imageN)
    return scores

def CollateBoundingBoxes(bboxs):
    output = []
    bboxs = np.array([b for b in bboxs if b[2] * b[3] > 20])
    while (len(bboxs) > 0):
        current = list(bboxs[0])
        current.append(0)
        removed = [0]
        for i in range(1, len(bboxs)):
            current_bbox = np.array(current[:4])
            comparing_bbox = bboxs[i]
            iou = IoUScore(current_bbox, comparing_bbox)
            if (iou > 0.05):
                new_bbox = ((comparing_bbox * current[4]) + comparing_bbox) / (current[4] + 1)
                current = [int(new_bbox[0]), int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3]), current[4] + 1]
                removed.append(i)
        bboxs = np.delete(bboxs, removed, axis=0)
        output.append(current[:4])
    return output

def combined_dectectors(image, model, truths, imageN, overlay=False):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    magnitude, direction = sobel_information(frame_gray)
    circles = hough_circles(direction, thresholded_pixels(magnitude, 200), frame_gray.shape, 11)
    ellipses = []

    if (len([c for c in circles if c[3] > 40]) == 0 and len(circles) > 0):
        min_radius = 30 # Remember to add min radius to each of the radii afterwards
        print("Ellipse Detecting")
        ellipses = hough_ellipse(direction, thresholded_pixels(magnitude, 200), frame_gray.shape, 11, (min_radius, 100))

    if (overlay):
        overlayed = np.copy(image)
        for (y, x, r, c) in circles:
            overlayed = cv2.circle(overlayed, (x, y), r, (0, 0, 255), 1)
        for (y, x, r_a, r_b, c) in ellipses:
            overlayed = cv2.ellipse(overlayed, (x, y), (r_a, r_b), 0, 0, 360, (0, 0, 255), 1)
        cv2.imwrite("circles_detected.jpg", overlayed)

            
    dartboards = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=8, flags=0, minSize=(30,30), maxSize=(300,300))

    detections = []
    # For each circle create bounding box and compute IoU score with viola jones detections

    scores = np.zeros(len(dartboards))
    circles_consumed = [[] for _ in range(len(dartboards))]
    for c in range(len(circles)):
        (y, x, r, confidence) = circles[c]
        bbox = [x - r, y - r, 2 * r, 2 * r]
        for i in range(len(dartboards)):
            score = IoUScore(bbox, dartboards[i])
            scores[i] += score
            if (score > 0):
                circles_consumed[i].append(c)
    

    for i in range(len(scores)):
        if (scores[i] > 0.1):
            detections.append(dartboards[i])
            circles = [circles[c] for c in range(len(circles)) if c not in circles_consumed[i]]

    for c in range(len(circles)):
        (y, x, r, confidence) = circles[c]
        if (confidence >= 30):
            bbox = [x - r, y - r, 2 * r, 2 * r]
            detections.append(bbox)
    
    for e in range(len(ellipses)):
        (y, x, r_a, r_b, confidence) = ellipses[e]
        if (confidence >= 150):
            bbox = [x - r_a, y - r_b, 2 * r_a, 2 * r_b]
            detections.append(bbox)

    detections.extend(DetectDartboardsFromTriangles(image))
    detections = CollateBoundingBoxes(np.array(detections))
    for d in detections:
        start_point = (d[0], d[1])
        end_point = (d[0] + d[2], d[1] + d[3])
        colour = (0, 255, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, colour, thickness)
    
    DrawGroundTruth(image, truths, imageN)
    scores = ScorePredictions(detections, truths[imageN])
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

def ClassifyMultiplePhotos(name_range, folder='Dartboard/', prefix='dart', ext='jpg', circles=False, ellipse=False):
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

        if (circles):
            scores.append(combined_dectectors(frame, model, ground_truths, i))
        else:
            scores.append(ViolaJonesDetector(model, frame, ground_truths, i))
        cv2.imwrite(f"Detected/{prefix}{i}.{ext}", frame)
    PrettyPrintScores(scores)

def TestContour(filepath):
    frame = cv2.imread(filepath, 1)
    dartboards = DetectDartboardsFromTriangles(frame)
    for d in dartboards:
        start_point = (d[0], d[1])
        end_point = (d[0] + d[2], d[1] + d[3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    cv2.imwrite("countour_test.jpg", frame)

def ClassifySinglePhoto(filepath, circles=True, overlay=False):
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

    if (circles):
        combined_dectectors(frame, model, ground_truths, imageNo, overlay)
    else:
        ViolaJonesDetector(model, frame, ground_truths, imageNo)
    cv2.imwrite(f"Detected/{filename}", frame)

imageName = args.name
if (args.name != ""):
    #TestContour(imageName)
    ClassifySinglePhoto(imageName, True, True)
else:
    ClassifyMultiplePhotos((0, 15), circles=True)
