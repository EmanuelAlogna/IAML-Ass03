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

capture = cv2.VideoCapture(INPUT_FILEPATH)

# just use the first frame for test purposes
retval, frame = capture.read()


# script for slidind window from "Sliding Windows for Object Detection with Python and OpenCV"
# (see https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
def sliding_window(image, stepSize=(16,16), windowSize=(64,64)):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


detected_cars = []
org_height, org_width = frame.shape[:2] 

scaled_frames = list(pyramid_gaussian(frame, downscale=1.5))

# while True:
#     cv2.imshow("Scale Test", scaled_frames[10])
#     if cv2.waitKey(15) & 0xFF == ord("q"):
#         break


# loop over every image scale
for image_scaled in scaled_frames:
    # loop over window in the image
    scaled_height, scaled_width = image_scaled.shape[:2]
    print((scaled_height, scaled_width))
    for (x, y, image_window) in sliding_window(image_scaled):

        print(x)
        print(y)
        image_window = image_window.astype(np.uint8)

        hogList = []

        # Compute the HOG
        As301_classifier.computeHOG([image_window],hogList, size=(64,64))

        # load the SVM classifier model
        classifier = joblib.load("./inputs/svm_model_weights.pkl")

        hog_featuers = hogList[0].reshape(1,1764)
        prediction = classifier.predict(hog_featuers)

        print(prediction)
        sys.exit()

#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->

capture.release()
