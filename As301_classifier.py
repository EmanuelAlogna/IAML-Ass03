#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : As301_classifier.py                                      -->
#<!-- Description: Script to train a car detector based on binary classifier-->
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
import random

from glob import glob

########################################################################

def loadDataset(dataset):
    """
    This function load all images from a dataset and return a list of Numpy images.
    """
    # List of images.
    images = []

    # Read all filenames from the dataset.
    for filename in dataset:
        # Read the input image.
        image = cv2.imread(filename)

        # Add the current image on the list.
        if image is not None:    
            images.append(image)
        else:
            print("Could not read file: {}".format(filename))
            sys.exit()

    # Return the images list.
    return images

def sampleNegativeImages(images, negativeSample, size=(64, 64), N=200):
    """
    The dataset has several images of high resolution without cars,
    i.e. called here as negative images. This function select "N" 64x64 negative
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

    return 

def computeHOG(images, hogList, size=(64, 128)):
    """
    This function computes the Histogram of Oriented Gradients (HOG) of each
    image from the dataset.
    [Code from Exercise 10 solution. Could be used for a SVM with HOG Features]
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


# Folder where the dataset images are saved.
folder = "./inputs"

# Dataset filenames.
positiveFile = glob("%s/cars/*.png" % folder)
negativeFile = glob("%s/non-cars/*.png" % folder)

# Vectors used to train the dataset.
positiveList = []
negativeList = []
negativeSample = []
labels = []

# Load the INRIA dataset.
positiveList = loadDataset(positiveFile)
negativeList = loadDataset(negativeFile)

print(str(len(positiveList)))
print(str(len(negativeList)))

# Get a sample of negative images. (returns list in negativeSample)
sampleNegativeImages(negativeList, negativeSample)

#<!--------------------------------------------------------------------------->
#<!--                            YOUR CODE HERE                             -->
#<!--------------------------------------------------------------------------->



#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->
