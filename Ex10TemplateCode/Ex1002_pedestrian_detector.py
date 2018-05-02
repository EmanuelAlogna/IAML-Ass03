#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : Ex1002_pedestrian_detector.py                            -->
#<!-- Description: Script to detect pedestrians using HOG and SVM           -->
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
        image = cv2.imread(filename)
    
        # Read the next image filename.
        filename = file.readline()
    
        # Add the current image on the list.
        if image is not None:    
            images.append(image)

    # Return the images list.
    return images

# Create the HOG descriptor and import the HOG feature.
feature = np.load("./outputs/feature.npy")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(feature)

# Dataset filename.
positiveFile = "./inputs/Test/pos.lst"

# Load the INRIA dataset.
positiveList = loadDataset(positiveFile)

# --- ADD CODE HERE TO READ A VIDEO INTO IMAGES ---

# Detects objects of different sizes in the input image
rectangles, _ = hog.detectMultiScale(image)

# Draw the detetcted pedestrians
for (x, y, w, h) in rectangles:
    cv2.rectangle(image, (x,y), (x + w, y + h), (0,0,255))

# --- ADD CODE TO DISPLAY THE RESULT IMAGE ---