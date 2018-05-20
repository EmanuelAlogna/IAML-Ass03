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
import pandas as pd
import random
import sklearn
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import sys

import imutils
import matplotlib.pyplot as plt

from glob import glob
from random import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist



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


def samplePositiveImages(images, positiveSample, size=(64, 64), N=200):
    """
    The dataset has not enough positive images, so we'll increase it by generating new positive
    images by, first, using linear transormation (rotation and reflection) on the
    available positive subset
    """

    for image in images:
    
        rotated = imutils.rotate_bound(image, random.randint(-15,15))
        
        h, w, channels = rotated.shape
        cropped_img = rotated[w//2 - 64//2:w//2 + 64//2, h//2 - 64//2:h//2 + 64//2]
        
        positiveSample.append(image);
        positiveSample.append(cropped_img)
        positiveSample.append(np.fliplr(image))
        positiveSample.append(np.fliplr(cropped_img))
    

    return

def getY(positiveImages, negativeImages):
    
    sizePositive = len(positiveImages)
    sizeNegative = len(negativeImages)

    labels = []
    
    for x in range (0, sizePositive):
        labels.append(1)

    for x in range (0, sizeNegative):
        labels.append(-1)

    shuffle(labels)
    return labels;

#<!--------------------------------------------------------------------------->
#<!--------------------------------------------------------------------------->
#<!--------------------------------------------------------------------------->
#<!--------------------------------------------------------------------------->



def main():

    # Folder where the dataset images are saved.
    folder = "./inputs"

    # Dataset filenames.
    positiveFile = glob("%s/cars/*.png" % folder)
    negativeFile = glob("%s/non-cars/*.png" % folder)

    # Vectors used to train the dataset.
    positiveList = []
    negativeList = []
    negativeSample = []
    positiveSample = []
    labels = []
    X = []

    # As 3.02. (a) : Load our car images dataset.
    positiveList = loadDataset(positiveFile)
    negativeList = loadDataset(negativeFile)
    
    #print("Initial size of car set: {0} \t\t (dim: {1})".format(len(positiveList), positiveList[0].shape))
    #print("Initial size of non-car set: {0} \t\t (dim: {1})".format(len(negativeList), negativeList[0].shape))


    # As 3.02. (b) : Get a sample of negative images. (returns list in negativeSample)
    sampleNegativeImages(negativeList, negativeSample, size=(64,64), N=200)
    
    samplePositiveImages(positiveList, positiveSample, size=(64,64), N=200)
    
  
    print("Size of non-car sample set: {0} \t (dim: {1})".format(len(negativeSample), negativeSample[0].shape))
    

    #-----------------------------------------------------------#
    #                                                           #
    # Classification Model using Convolutionary neural network  #
    #                                                           #
    #-----------------------------------------------------------#

    np.random.seed(123)
    y = getY(positiveSample, negativeSample)

    for image in positiveSample:
        X.append(image)

    for image in negativeSample:
        X.append(image)

    shuffle(X)

    print(len(X))
    print(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=1, shuffle=True)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print (X_train.shape)

    print("Count of non-car/car in training data")
    print(y_train.shape[0])
    print("Count of non-car/car in test data")
    print(y_test.shape[0])

    # the final preprocessing step for the input data is to convert our data
    # type to float 32 and normalize our data values to the range[0, 1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # preprocessing class labels for Keras
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    print(y_train.shape)

    model = Sequential()

    # 32 corresponds to the number of convolution filters to use
    # 3 corresponds to the numbers of rows in each convolution kernel
    # 3 corresponds to the number of columns in each convolution kernel
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))

    print (model.output_shape)
    
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))  # this layer is important because it prevents overfitting

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax')) #output layer with size of 2 (2 classes)

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # fit model on training data
    model.fit(X_train, y_train,
              batch_size=32, nb_epoch=2, verbose=1)

    # evaluate model on test data
    score = model.evaluate(X_test, y_test, verbose=0)

    print(score)

    model.save('./outputs/datamodel.h5')

#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->

# put executing code in main, so that the defined functions
# can be imported in a separate script without executing
# the code
if __name__ == "__main__":
    main()

