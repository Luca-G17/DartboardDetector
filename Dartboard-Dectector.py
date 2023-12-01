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
    cv2.imwrite("grad_magnitude.jpg", (grad_magnitude > 240) * 255)
    grad_direction = np.arctan2(grad_y, grad_x + 1e-10)
    return (grad_magnitude, grad_direction)

def thresholded_pixels(image, T):
    return np.argwhere(image > T)

def hough_circles(grad_direction, thresholded_grad_mag, shape, T):
    height, width = shape
    max_radius = 50
    H = np.zeros((height + max_radius, width + max_radius, max_radius))
    for pixel in thresholded_grad_mag:
        (y, x) = pixel
        for r in range(10, max_radius):
            x_comp = int(r * math.cos(grad_direction[y][x]))
            y_comp = int(r * math.sin(grad_direction[y][x]))
            H[y + y_comp][x + x_comp][r] += 1
            H[y - y_comp][x - x_comp][r] += 1  
    return np.argwhere(H > T)

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
    for pixel in thresholded_grad_mag:
        (y, x) = pixel
        for r_a in range(0, max_radius - min_radius):
            for r_b in range(0, max_radius - min_radius): # Semi-minor axis is always less than the semi-major axis
                alpha = grad_direction[y][x]
                x_delta = int((r_a + min_radius) * math.cos(alpha))
                y_delta = int((r_b + min_radius) * math.sin(alpha))
                H[y + y_delta][x + x_delta][r_a][r_b] += 1
                H[y - y_delta][x - x_delta][r_a][r_b] += 1

    es = np.argwhere(H > T)
    remaining = np.copy(es)
    averages = []
    H = (H <= T) * 0
    cluster_radius = 20
    while(len(remaining) > 0):
        average = remaining[0]
        (ay, ax, ar_a, ar_b) = average
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
                new_coords = ((acoords * an) + coords) / (an + n)
                new_radii = ((aradii * an) + radii) / (an + n)
                average = [new_coords[0], new_coords[1], new_radii[0], new_radii[1], an + n] 
                removed.append(i)
        remaining = np.delete(remaining, removed)
        averages.append(average)

    return averages

def semi_minor_axis(p1, p2, p3, a, d2):
    f2 = np.dot(p3 - p2, p3 - p2)
    a2 = a ** 2
    denom = (4 * a2 * float(d2))
    if (denom == 0):
        return None 
    cos_t2 = ((a2 + d2 - f2) ** 2) / denom
    sin_t2 = 1 - cos_t2
    denom = (a2 * -d2 * cos_t2)
    if (denom == 0):
        return None
    b2 = a2 * d2 * sin_t2 / denom
    if (b2 <= 0):
        return None 
    return math.sqrt(b2) 
    
def better_hough_ellipse(grad_direction, thresholded_grad_map, shape, T):
    accumulator = np.full(100, 0)
    max_major_axis = 100
    ellipses = []
    print(len(thresholded_grad_map))
    for p1 in thresholded_grad_map:
        print(p1)
        (y1, x1) = p1
        for p2 in thresholded_grad_map:
            (y2, x2) = p2
            if (np.array_equal(p1, p2)):
                continue
            dist2 = np.dot(p1 - p2, p1 - p2)
            if (dist2 > 200 ** 2):
                continue
            centre_x = (x1 + x2) / 2
            centre_y = (y1 + y2) / 2
            centre = np.array([centre_y, centre_x])
            major_axis = math.sqrt(dist2) / 2
            angle = np.arctan2((y2 - y1), (x2 - x1))
            for p3 in thresholded_grad_map:
                (y, x) = p3
                if (np.array_equal(p3, p1) or np.array_equal(p3, p2)):
                    continue
                dist2 = np.dot(p3 - centre, p3 - centre)
                if (dist2 > max_major_axis ** 2):
                    continue
                b = semi_minor_axis(p1, p2, p3, major_axis, dist2)
                if b is None:
                    continue
                accumulator[int(b)] += 1
            voted = np.argmax(accumulator)
            if (accumulator[voted] > T):
                ellipses.append((centre_y, centre_x, major_axis, voted, angle))
    return ellipses

def LineLineIntersection(a0, b0, c0, a1, b1, c1):
    determinant = (a0 * b1) - (a1 * b0)
    if (determinant == 0):
        return None
    else:
        x = ((c0 * b1) + (c1 * b0)) / determinant
        y = ((a0 * c1) + (a1 * c0)) / determinant
        return np.array([x, y])

def TwoDLineIntersectionFromPoints(A, B, C, D):
    a0 = B[1] - A[1]
    b0 = A[0] - B[0]
    c0 = (a0 * A[0]) + (b0 * A[1])

    a1 = D[1] - C[1]
    b1 = C[0] - D[0]
    c1 = (a1 * C[0]) + (b1 * C[1])
    return LineLineIntersection(a0, b0, c0, a1, b1, c1)

# y=mx+c
def TwoDLineIntersectionFromGradient(m0, c0, m1, c1):
    # ax+by=c
    a0 = 1
    b0 = 1 / m0
    c0 = c0 / m0

    a1 = 1
    b1 = 1 / m1
    c1 = c1 / m1
    return LineLineIntersection(a0, b0, c0, a1, b1, c1)

def PointAndAngleToLine(p, theta):
    m = math.tan(theta)
    c = p[1] - (m * p[0])
    return [m, c]

def MidPoint(X1, X2):
    return (X1 + X2) / float(2.0)

def ThreePointsToEllipseCentre(p0, a0, p1, a1, p2, a2):
    # Tangent lines = T1, T2, T3
    T1 = PointAndAngleToLine(p0, a0 + (math.pi / 2))
    T2 = PointAndAngleToLine(p1, a1 + (math.pi / 2))
    T3 = PointAndAngleToLine(p2, a2 + (math.pi / 2))
    T12 = TwoDLineIntersectionFromGradient(T1[0], T1[1], T2[0], T2[1])
    T23 = TwoDLineIntersectionFromGradient(T2[0], T2[1], T3[0], T3[1])
    if (T12 is None or T23 is None):
        return None
    M12 = MidPoint(p0, p1)
    M23 = MidPoint(p1, p2)
    return TwoDLineIntersectionFromPoints(T12, M12, T23, M23)

def SolveQuadratic(a, b, c, plus=True):
    if (a == 0):
        return None
    discriminant = (b ** 2) - (4 * a * c)
    if (discriminant < 0):
        return None
    
    return (-b + math.sqrt(discriminant)) / (2 * a)

def ThreePointsToEllipseRadii(X0, X1, X2, centre):
    X0 = X0 - centre
    X1 = X1 - centre
    X2 = X2 - centre
    P0 = np.array([X0[0] ** 2, 2 * X0[0] * X0[1], X0[1] ** 2])
    P1 = np.array([X1[0] ** 2, 2 * X1[0] * X1[1], X1[1] ** 2])
    P2 = np.array([X2[0] ** 2, 2 * X2[0] * X2[1], X2[1] ** 2])
    A = np.vstack((P0, P1, P2))
    if (np.linalg.matrix_rank(A) != A.shape[0]):
        return None
    Ainv = np.linalg.inv(A)
    [alpha, beta, gamma] = np.sum(Ainv, axis=1)
    x = SolveQuadratic(alpha, -2 * beta * centre[1], gamma)
    y = SolveQuadratic(gamma, -2 * beta * centre[0], alpha)
    if (x == None or y == None):
        return None
    return np.array([abs(centre[0] - x), abs(centre[1] - y)])

class Ellipse:
    def __init__(self, indices, radii, centre):
        self.indices = indices
        self.radii = radii
        self.centre = centre
        self.score = 0
        self.n = 1

    def Average(self, c, r, indices):
        self.radii = (r + (self.n * self.radii)) / (self.n + 1)
        self.centre = (c + (self.n * self.centre)) / (self.n + 1)
        self.n += 1
        self.indices.extend(indices)

def RandomWithinRange(i0, ps, T=50):
    i = random.randint(0, len(ps) - 1)
    while (np.dot(ps[i0] - ps[i], ps[i0] - ps[i]) > T ** 2 or i == i0):
        i = random.randint(0, len(ps) - 1)

    i2 = random.randint(0, len(ps) - 1)
    while (np.dot(ps[i0] - ps[i2], ps[i0] - ps[i2]) > T ** 2 or i2 == i0 or i2 == i):
        i2 = random.randint(0, len(ps) - 1)
    return np.array([i0, i, i2])
                
def randomized_hough_ellipse(grad_direction, thresholded_grad_mag, ellipses_n=20, iters=1000):
    # Accumulator:
    # For each ellipse, store a list of generating points indices
    # semi-major/minor axis, centre coords
    ellipses = []
    while (len(thresholded_grad_mag) > 3 and len(ellipses) < ellipses_n):
        accumulator = []
        for i in range(iters):
            # Find potential ellipse
            pointIndices = RandomWithinRange(random.randint(0, len(thresholded_grad_mag) - 1), thresholded_grad_mag)
            # Find centre coords
            points = np.fliplr(np.array(thresholded_grad_mag[pointIndices]))
            angles = []
            for point in points:
                angles.append(grad_direction[point[1]][point[0]])
            centre = ThreePointsToEllipseCentre(points[0], angles[0], points[1], angles[1], points[2], angles[2])
            if centre is None:
                continue

            radii = ThreePointsToEllipseRadii(points[0], points[1], points[2], centre)
            if radii is None:
                continue
            # Measure similarity between this ellipse and all the other ellipses in the accumulator
            found = False
            for ellipse in accumulator:
                if EllipseSimiliarity(centre, radii, ellipse.centre, ellipse.radii):
                    found = True
                    ellipse.score += 1
                    #ellipse.Average(centre, radii, pointIndices)
            if (found == False):
                accumulator.append(Ellipse(pointIndices, radii, centre))
        if (len(accumulator) > 0):
            ellipse = max(accumulator, key=lambda e:e.score)
            ellipses.append(ellipse)
            thresholded_grad_mag = np.delete(thresholded_grad_mag, ellipse.indices, 0)
        
    return [(e.centre[0], e.centre[1], e.radii[0], e.radii[1]) for e in ellipses]

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
    magnitude, direction = sobel_information(frame_gray)
    #ellipses = better_hough_ellipse(direction, thresholded_pixels(magnitude, 250), frame_gray.shape, 10)
    # ellipses = randomized_hough_ellipse(direction, thresholded_pixels(magnitude, 200))
    min_radius = 45
    ellipses = hough_ellipse(direction, thresholded_pixels(magnitude, 240), frame_gray.shape, 12, (min_radius, 100))
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
