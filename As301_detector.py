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
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
import sys

import As301_classifier

########################################################################

INPUT_FILEPATH = "./inputs/videos/Cars_01.mov"

########################################################################

# We are going to apply both Image Pyramids and Sliding Windows 
# for our car detector and then use our classification model on the
# image patches

########################################################################

# script for sliding window from "Sliding Windows for Object Detection with Python and OpenCV"
# (see https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
def sliding_window(image, stepSize=(24,24), windowSize=(64,64)):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# function returns a image where rectangles mark the detected cars
def detectCars(frame):

    org_height, org_width = frame.shape[:2] 

    # store the frame in different dimensions 
    # and begin with the lowest resolution
    scaled_frames = list(pyramid_gaussian(frame, downscale=1.5, max_layer=8))
    scaled_frames = list(reversed(scaled_frames))

    detected_cars = []

    # loop over every image scale
    for image_scaled in scaled_frames:

        # loop over window in the image
        scaled_height, scaled_width = image_scaled.shape[:2]
        SCALING_FACTOR = (org_height / scaled_height + org_width / scaled_width) / 2.0
        # print((scaled_height, scaled_width))
        # print("Scaling Factor : {}".format(SCALING_FACTOR))

        if scaled_height < 300 or scaled_width < 400:
            continue

        if scaled_height > 600 or scaled_width > 1100:
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
                # do not add rectangles that are enclosed by 
                # a previously detected rectangle

                BORDER_PIXELS = 20
                isOverlapping = False

                for (prev_x,prev_y,prev_scale) in detected_cars:
                    prev_bounds_x = (int(prev_x * prev_scale), int((prev_x + 64) * prev_scale) ) 
                    prev_bounds_y = (int(prev_y * prev_scale), int((prev_y + 64) * prev_scale) ) 

                    overlapCheck = (
                                    int(x*SCALING_FACTOR) > prev_bounds_x[0] - BORDER_PIXELS and
                                    int((x + 64) * SCALING_FACTOR) < prev_bounds_x[1] + BORDER_PIXELS and
                                    int(y*SCALING_FACTOR) > prev_bounds_y[0] - BORDER_PIXELS and
                                    int((y + 64) * SCALING_FACTOR) < prev_bounds_y[1] + BORDER_PIXELS
                                    )

                    if overlapCheck:
                        isOverlapping = True
                        # print("image overlaps {0} , {1}".format(prev_bounds_x, x*SCALING_FACTOR))

                if not isOverlapping:
                    detected_cars.append((x,y,SCALING_FACTOR))

    res_image = frame.copy()
    for (x,y,scale) in detected_cars:
        cv2.rectangle(res_image, 
            (int(x * scale), int(y * scale)) , 
            (int((x+64)*scale),int((y+64)*scale)), (0,255,0),2)

    return res_image

# Setup Video

capture = cv2.VideoCapture(INPUT_FILEPATH)

# Get the video frame rate.
fps = int(round(capture.get(cv2.CAP_PROP_FPS)))

# Check if the fps variable has a correct value.
fps = fps if fps > 0 else 30

# Create an OpenCV window.
cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

# just use the first frame for test purposes
retval, frame = capture.read()
            
while True:
    # Capture frame-by-frame.
    retval, frame = capture.read()

    # Check if there is a valid frame.
    if not retval:
        break

    frame = detectCars(frame)

    # Resize the frame.
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Display the resulting frame.
    cv2.imshow("Video", frame)
    if cv2.waitKey(fps) & 0xFF == ord("q"):
        break


#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->

capture.release()
cv2.destroyAllWindows()