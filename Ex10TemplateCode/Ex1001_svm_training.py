#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : Ex1001_svm_training.py                                   -->
#<!-- Description: Script to train a pedestrian detector based on SVM       -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : narcizo[at]itu[dot]dk                                    -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 29/03/2018                                               -->
#<!-- Change     : 29/03/2018 - Creation of this script                     -->
#<!-- Review     : 29/03/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018032901 $"

########################################################################
import cv2
import numpy as np
import random
import sys
import os
cwd = os.getcwd() + "\\"

########################################################################

def loadDataset(dataset):
    """
    This function load all images from a dataset and return a list of Numpy images.
    """
    # List of images.
    images = []

    # Read the dataset file.
    file = open(dataset)
    filename = file.readline()

    # Read all filenames from the dataset.
    while (filename != ""):
        # Get the current filename.

        filename = (dataset.rsplit("/", 1)[0] + "/" +
                    filename.split("/", 1)[1].strip("\n"))

        # Read the input image.
        # ERROR: My image files in the dataset seem to be corrupted
        # Will add the code to the Ass3 Repo as template for our
        # car detector. There the images do not seem to be corrupted.
        image = cv2.imread(full_path)
        # C:\Users\Peter Mortimer\Desktop\IAML\Week10\Exercises_10_material\inputs\Train\pos

        if image is None:
            print("Could not read the file: {}".format(full_path))
    
        # Read the next image filename.
        filename = file.readline()
    
        # Add the current image on the list.
        if image is not None:    
            images.append(image)

    # Return the images list.
    return images

def getSVMDetector(svm):
    """
    This function calculates and returns the feature descriptor.
    """
    # Retrieves all the support vectors.
    sv = svm.getSupportVectors()

    # Retrieves the decision function.
    rho, _, _ = svm.getDecisionFunction(0)

    # Transpose the support vectors matrix.
    sv = np.transpose(sv)

    # Returns the feature descriptor.
    return np.append(sv, [[-rho]], 0)

def sampleNegativeImages(images, negativeSample, size=(64, 128), N=10):
    """
    INRIA Dataset has several images of different resolution without pedestrians,
    i.e. called here as negative images. This function select "N" 64x128 negative
    sub-images randomly from each original negative image.
    """
    # Initialize internal state of the random number generator.
    random.seed(1)

    # Final image resolution.
    w, h = size[0], size[1]

    # Read all images from the negative list.
    for image in images:

        for j in range(N):
            # random.random produced random number in [0,1) range
            y = int(random.random() * (len(image) - h))
            x = int(random.random() * (len(image[0]) - w))
            sample = image[y:y + h, x:x + w].copy()
            negativeSample.append(sample)

    return negativeSample

def computeHOG(images, hogList, size=(64, 128)):
    """
    This function computes the Histogram of Oriented Gradients (HOG) of each
    image from the dataset.
    """
    # Creates the HOG descriptor and detector with default parameters.
    hog = cv2.HOGDescriptor()

    # Read all images from the image list.
    for image in images:
    
        # Image resolution
        h, w = image.shape[:2]

        # Calculate HOG
        if w >= size[0] and h >= size[1]:

            # Region of Interest
            y = (h - size[1]) // 2
            x = (w - size[0]) // 2
            roi = image[y:y + size[1], x:x + size[0]].copy()

            # Compute HOG
            grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hogList.append(hog.compute(grayscale))

    return hogList


# Dataset filenames.
positiveFile = "./inputs/train/pos.lst"
negativeFile = "./inputs/train/neg.lst"

# Vectors used to train the dataset.
hogList = []
positiveList = []
negativeList = []
negativeSample = []
hardNegativeList = []
labels = []

# Load the INRIA dataset.
positiveList = loadDataset(positiveFile)
negativeList = loadDataset(negativeFile)

print(str(len(positiveList)))
print(str(len(negativeList)))

# Get a sample of negative images.
sampleNegativeImages(negativeList, negativeSample)

# Compute the Histogram of Oriented Gradients (HOG).
computeHOG(positiveList, hogList)
computeHOG(negativeSample, hogList)
if len(hogList) == 0:
    exit(0)

# Create the class labels, i.e. (+1) positive and (-1) negative.
[labels.append(+1) for _ in range(len(positiveList))]
[labels.append(-1) for _ in range(len(negativeSample))]

# Create an empty SVM model.
svm = cv2.ml.SVM_create()

# Define the SVM parameters.
# By default, Dalal and Triggs (2005) use a soft (C=0.01) linear SVM trained with SVMLight.
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
svm.setTermCriteria(criteria)
svm.setGamma(0)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)
svm.setC(0.01)
svm.setType(cv2.ml.SVM_EPS_SVR)

svm.train(np.array(hogList), cv2.ml.ROW_SAMPLE, np.array(labels))

# Create the HOG descriptor and detector with default params.
hog = cv2.HOGDescriptor()
hog.setSVMDetector(getSVMDetector(svm))

# Add the wrong identification sample for the second round of training
# (hard examples)
for image in negativeList:
    # Detects objects od different sizes in the input image
    rectangles, _ = hog.detectMultiScale(image)

    # Get the ROI in the false positive pedestrian
    for (x, y, w, h) in rectangles:
        roi = image[y:y + h, x:x + w]
        hardNegativeList.append(cv2.resize(roi, (64,128)))

# Compute the Histogram of Oriented Gradients (HOG)
computeHog(hardNegativeList, hogList)

# Update the class labels, i.e. (-1) hard negative.
# (these are added to the list with pos. and neg. images)
[labels.append(-1) for _ in range(len(hardNegativeList))]

# Train the SVM based on HOG Features
svm.train(np.array(hogList), cv2.ml.ROW_SAMPLE, np.array(labels))

# Save the HOG feature.
feature = getSVMDetector(svm)
np.save("./outputs/feature.npy", feature)
print("stored SVM weights")
