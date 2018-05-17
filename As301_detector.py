#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : As301_detector.py.py                                     -->
#<!-- Description: Script to detect cars using a binary classifier          -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : narcizo[at]itu[dot]dk                                    -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 24/04/2018                                               -->
#<!-- Change     : 24/04/2018 - Creation of this script                     -->
#<!-- Review     : 24/04/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018042401 $"

########################################################################
import cv2
import numpy as np
from pathlib import Path
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
import sys

import As301_classifier

########################################################################

INPUT_FILEPATH = Path("./inputs/videos/Cars_05.mov")
FILENAME = INPUT_FILEPATH.stem

########################################################################

# We are going to apply both Image Pyramids and Sliding Windows 
# for our car detector and then use our classification model on the
# image patches

########################################################################

# script for sliding window from "Sliding Windows for Object Detection with Python and OpenCV"
# (see https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
def sliding_window(image, stepSize=(16,16), windowSize=(64,64)):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# function returns a list of rectangles that were detected as cars
def detectCars(frame):

    org_height, org_width = frame.shape[:2] 

    # store the frame in different dimensions 
    # and begin with the lowest resolution
    scaled_frames = list(pyramid_gaussian(frame, downscale=1.5, max_layer=3))
    scaled_frames = list(reversed(scaled_frames))

    detected_cars = []
    rectangles = []

    # loop over every image scale
    for image_scaled in scaled_frames:

        # loop over window in the image
        scaled_height, scaled_width = image_scaled.shape[:2]
        SCALING_FACTOR = (org_height / scaled_height + org_width / scaled_width) / 2.0
        
        if scaled_height < 64 or scaled_width < 64:
            continue

        for (x, y, image_window) in sliding_window(image_scaled):

            if x > scaled_width - 64 or y > scaled_height - 64:
                continue

            # convert from float [0,1] range to integer [0,255] range
            image_window = image_window * 255
            image_window = image_window.astype(np.uint8)
            hogList = []

            # Compute the HOG
            As301_classifier.computeHOG([image_window],hogList, size=(64,64))

            # load the SVM classifier model
            classifier = joblib.load("./inputs/svm_model_weights.pkl")

            try:
                hog_featuers = hogList[0].reshape(1,1764)
                prediction = classifier.predict(hog_featuers)
            except IndexError:
                print("Caught an IndexError")
                print((x,y))
                print(hogList)
                sys.exit()

            # create a list of detected cars in the image
            if prediction == [1]:
                # As 3.02. (k) : resolve overlapping bounding boxes
                # do not add rectangles that are enclosed by 
                # a previously detected rectangle

                BORDER_PIXELS = 30
                isOverlapping = False

                currRect = (        int(x*SCALING_FACTOR),
                                    int(y*SCALING_FACTOR),
                                    int((x + 64) * SCALING_FACTOR),
                                    int((y + 64) * SCALING_FACTOR) )

                for (prev_x,prev_y,prev_scale) in detected_cars:

                    oldRect = (     int(prev_x * prev_scale),
                                    int(prev_y * prev_scale),
                                    int((prev_x + 64) * prev_scale),
                                    int((prev_y + 64) * prev_scale) )

                    if rectangleOverlap(rect1 = oldRect, rect2 = currRect, margin = BORDER_PIXELS):
                        isOverlapping = True
                        break

                if not isOverlapping:
                    detected_cars.append((x,y,SCALING_FACTOR))

    res_image = frame.copy()
    for (x,y,scale) in detected_cars:
        rectangles.append((
                    int(x * scale),
                    int(y * scale),
                    int(64 * scale),
                    int(64 * scale)
            ))

    return rectangles

# uses background substraction to find ROIs for our classifier
def backgroundDetection(frame):
    rectangles = []
    processed = fgbg.apply(frame)
    _, contours, hierarchy = cv2.findContours(processed, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area < 500):
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        rectangles.append((x,y,w,h))

    return rectangles

# returns True if (x1,y1,a1,b1) encloses (x2,y2,a2,b2) with a certain 
# (x,y) = coords of the top left corner point
# (a,b) = coords of the bottom right corner point
# margin of pixel allowance
def rectangleOverlap(rect1 = (0,0,0,0), rect2 = (0,0,0,0), margin=0):
    x1, y1, a1, b1 = rect1
    x2, y2, a2, b2 = rect2

    overlapCheck = (
                    x2 > x1 - margin and
                    a2 < a1 + margin and
                    y2 > y1 - margin and
                    b2 < b1 + margin
                    )

    return overlapCheck 

# Setup Video

capture = cv2.VideoCapture(str(INPUT_FILEPATH))

# Get the video frame rate.
fps = int(round(capture.get(cv2.CAP_PROP_FPS)))

# Check if the fps variable has a correct value.
fps = fps if fps > 0 else 30

# Create an OpenCV window.
cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
fgbg = cv2.createBackgroundSubtractorMOG2()
            
while True:
    # Capture frame-by-frame.
    retval, frame = capture.read()

    # Check if there is a valid frame.
    if not retval:
        break

    # Resize the frame.
    scaleX, scaleY = (0.5,0.5)
    frame = cv2.resize(frame, (0, 0), fx=scaleX, fy=scaleY)

    bgRectangles = backgroundDetection(frame)

    # remove too small rectangles
    bgRectangles = [(x,y,w,h) for (x,y,w,h) in bgRectangles if (w*h > 1500 and w*h < 65000)]
    # remove rectangles that aren't square
    bgRectangles = [(x,y,w,h) for (x,y,w,h) in bgRectangles if ((w/h > 0.2) if h > w else (h/w > 0.2))]

    bg_rect_overlap_free = []

    # remove overlapping rectangles
    for (x1,y1,w1,h1) in bgRectangles:
        isOverlapping = False
        for (x2,y2,w2,h2) in bgRectangles:
            if (
                (x1,y1,w1,h1) != (x2,y2,w2,h2) and 
                rectangleOverlap((x2,y2,x2+w2,y2+h2), (x1,y1,x1+w1,y1+h1), margin=50)
                ):
                isOverlapping = True
                break
        if not isOverlapping:
            bg_rect_overlap_free.append((x1,y1,w1,h1))

    detectedRect = []

    # (x,y,w,h)
    for (x,y,w,h) in bg_rect_overlap_free:
        PIXEL_BOUND = 40
        bound_x = PIXEL_BOUND if x >= PIXEL_BOUND else x
        bound_y = PIXEL_BOUND if y >= PIXEL_BOUND else y
        # print((y-bound_y,y+h+bound_y,x-bound_x,x+w+bound_x))
        cv2.rectangle(frame, (x-bound_x, y-bound_y), (x+w+bound_x, y+h+bound_y), (0,0,255), 2 )
        detections = detectCars(frame[y-bound_y:y+h+bound_y,x-bound_x:x+w+bound_x,:])
        detections = [(x1+(x-bound_x),y1+(y-bound_y),w1,h1) for (x1,y1,w1,h1) in detections]
        detectedRect += detections

    # print("Detections before overlap {0}".format(len(detectedRect)))
    detect_count = 0
    for (x1,y1,w1,h1) in detectedRect:
        isOverlapping = False
        for (x2,y2,w2,h2) in detectedRect:
            if (
                (x1,y1,w1,h1) != (x2,y2,w2,h2) and 
                rectangleOverlap((x2,y2,x2+w2,y2+h2), (x1,y1,x1+w1,y1+h1), margin=50)
                ):
                isOverlapping = True
                break

        if not isOverlapping:
            detect_count += 1
            cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

    # print("Detections after overlap {0}".format(detect_count))

    # Display the resulting frame.
    cv2.imshow("Video", frame)
    if cv2.waitKey(fps) & 0xFF == ord("q"):
        break   


#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->

capture.release()
cv2.destroyAllWindows()