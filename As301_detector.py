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
from enum import Enum
import keras
from keras.models import load_model
import numpy as np
from pathlib import Path
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
import sys
import time

import As301_classifier

########################################################################

INPUT_FILEPATH = Path("./inputs/videos/Cars_05.mov")
FILENAME = INPUT_FILEPATH.stem

# Setup Video

RECORD_VIDEO = True

if RECORD_VIDEO:
    print("Recording a video of " + FILENAME + ".mov")

capture = cv2.VideoCapture(str(INPUT_FILEPATH))

# Get the video frame rate.
fps = int(round(capture.get(cv2.CAP_PROP_FPS)))
w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)

# Check if the fps variable has a correct value.
fps = fps if fps > 0 else 30

frame_count = 0
isColor = True
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
if RECORD_VIDEO:
    record  = cv2.VideoWriter("outputs/" + FILENAME + "_record.mov",
                      fourcc, fps, (w, h), isColor)

# Setup Classifiers

class Classifier(Enum):
    SVM = 0
    CNN = 1 

# pre-load all models
CLF_SVM = joblib.load("./inputs/svm_model_weights_rbf.pkl")
CLF_CNN = load_model("./outputs/datamodel30epochs.h5")


########################################################################

# We are going to apply both Image Pyramids and Sliding Windows 
# for our car detector and then use our classification model on the
# image patches

########################################################################

# script for sliding window from "Sliding Windows for Object Detection with Python and OpenCV"
# (see https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
def sliding_window(image, stepSize=(8,8), windowSize=(64,64)):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# function returns a list of rectangles that were detected as cars
def detectCars(frame, model = Classifier.SVM):

    org_height, org_width = frame.shape[:2] 

    # store the frame in different dimensions 
    # and begin with the lowest resolution
    scaled_frames = list(pyramid_gaussian(frame, downscale=1.5, max_layer=2))
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

        windows = list(sliding_window(image_scaled))

        windows = [w for w in windows if (w[0] <= scaled_width - 64 and w[1] <= scaled_height - 64)]

        x = [w[0] for w in windows]
        y = [w[1] for w in windows]
        image_window = np.array([w[2] for w in windows])

        # convert from float [0,1] range to integer [0,255] range
        image_window = image_window * 255
        image_window = image_window.astype(np.uint8)

        predictions = []

        if model == Classifier.SVM:

            hogList = []

            # Compute the HOG
            As301_classifier.computeHOG(image_window,hogList, size=(64,64))

            try:
                hog_features = np.array(hogList)
                num_patches, num_hog_features = hog_features.shape[:2]
                hog_features = hog_features.reshape((num_patches,num_hog_features))
                predictions = CLF_SVM.predict(hog_features)
            except IndexError:
                print("Caught an IndexError")
                print((x,y))
                print(hogList)
                sys.exit()

        elif model == Classifier.CNN:

            # TODO: don't forget to scale the input into [0,1] range from [0,255]

            predictions = CLF_CNN.predict_classes(np.array(image_window))

        else:
            raise Exception("Did not specify a valid model.")

        # create a list of detected cars in the image

        for idx, pred in enumerate(predictions):
            if pred == 1:
                detected_cars.append((x[idx],y[idx],SCALING_FACTOR))   

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

# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
# (code from https://stackoverflow.com/questions/37847923/combine-overlapping-rectangles-python)
def non_max_suppression_fast(boxes, overlapThresh):
   # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float") 
    # initialize the list of picked indexes   
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# Create an OpenCV window.
if not RECORD_VIDEO:
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

fgbg = cv2.createBackgroundSubtractorMOG2()

# measure the frame by frame calculation performance
frame_times = []
            
while True:
    # Capture frame-by-frame.
    retval, frame = capture.read()
    time_diff = time.time()

    # Check if there is a valid frame.
    if not retval:
        break

    # Resize the frame.
    scaleX, scaleY = (0.5,0.5)
    frame = cv2.resize(frame, (0, 0), fx=scaleX, fy=scaleY)

    bgRectangles = backgroundDetection(frame)

    # remove too small rectangles
    bgRectangles = [(x,y,w,h) for (x,y,w,h) in bgRectangles if (w*h > 700 and w*h < 65000)]
    # remove rectangles that aren't square
    bgRectangles = [(x,y,w,h) for (x,y,w,h) in bgRectangles if ((w/h > 0.2) if h > w else (h/w > 0.2))]

    bg_rect_overlap_free = []

    # remove overlapping rectangles
    for (x1,y1,w1,h1) in bgRectangles:
        isOverlapping = False
        for (x2,y2,w2,h2) in bgRectangles:
            if (
                (x1,y1,w1,h1) != (x2,y2,w2,h2) and 
                rectangleOverlap((x2,y2,x2+w2,y2+h2), (x1,y1,x1+w1,y1+h1), margin=20)
                ):
                isOverlapping = True
                break
        if not isOverlapping:
            bg_rect_overlap_free.append((x1,y1,w1,h1))

    detectedRect = []

    # (x,y,w,h)
    for (x,y,w,h) in bg_rect_overlap_free:
        PIXEL_BOUND = 20
        bound_x = PIXEL_BOUND if x >= PIXEL_BOUND else x
        bound_y = PIXEL_BOUND if y >= PIXEL_BOUND else y
        # cv2.rectangle(frame, (x-bound_x, y-bound_y), (x+w+bound_x, y+h+bound_y), (0,0,255), 2 )
        detections = detectCars(frame[y-bound_y:y+h+bound_y,x-bound_x:x+w+bound_x,:], model=Classifier.SVM)
        detections = [(x1+(x-bound_x),y1+(y-bound_y),w1,h1) for (x1,y1,w1,h1) in detections]
        detectedRect += detections

    # print("Detections before overlap {0}".format(len(detectedRect)))

    # convert from (x,y,w,h) to (x1,y1,x2,y2) for non-maximum suppression
    detectedRect = np.array([(x,y,x+w,y+h) for (x,y,w,h) in detectedRect])

    detectedRect = non_max_suppression_fast(detectedRect, 0.1)

    # print("Detections after overlap {0}".format(len(detectedRect)))

    for (x1,y1,x2,y2) in detectedRect:
        cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0), 2)

    # detect_count = 0
    # for (x1,y1,w1,h1) in detectedRect:
    #     isOverlapping = False
    #     for (x2,y2,w2,h2) in detectedRect:
    #         if (
    #             (x1,y1,w1,h1) != (x2,y2,w2,h2) and 
    #             rectangleOverlap((x2,y2,x2+w2,y2+h2), (x1,y1,x1+w1,y1+h1), margin=10)
    #             ):
    #             isOverlapping = True
    #             break

    #     if not isOverlapping:
    #         detect_count += 1
    #         cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

    # print("Detections after overlap {0}".format(detect_count))

    time_diff = time.time() - time_diff

    frame_times.append(time_diff)

    # Display the resulting frame.
    if RECORD_VIDEO == True:
        record.write(frame)
        if frame_count % 30 == 0:
            print("Processed {0} frames \t({1} seconds of video)".format(frame_count, frame_count//30))
            print("Average processing time for one frame {0}".format(str(np.mean(np.array(frame_times)))))
    else:
        cv2.imshow("Video", frame)
        if cv2.waitKey(fps) & 0xFF == ord("q"):
            break   

    frame_count += 1


#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->
if RECORD_VIDEO:
    record.release()

capture.release()
cv2.destroyAllWindows()