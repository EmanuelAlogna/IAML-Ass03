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
import os
import pandas as pd
import random
import sklearn
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import sys

import imutils

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

    resizedImages = []
    
    for image in images:
        res = cv2.resize(image, dsize=(1728, 1152), interpolation=cv2.INTER_CUBIC)
        resizedImages.append(res)

    for image in resizedImages:
        images.append(image)

    # Read all images from the negative list.

    i = 0
    for image in images:

        if i > 4:
            N = 100
        for j in range(N):
            # random.random produced random number in [0,1) range
            y = int(random.random() * (len(image) - h))
            x = int(random.random() * (len(image[0]) - w))
            sample = image[y:y + h, x:x + w].copy()
            negativeSample.append(sample)

            # Create Afine transform
            afine_tf = tf.AffineTransform(shear = random.uniform(-0.2,0.2))
            # Apply transform to image data
            shearedImage = tf.warp(sample, inverse_map=afine_tf)
            negativeSample.append(shearedImage)
        i = i + 1

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
            
    supportList = []
    for img in positiveSample:
        supportList.append(img)

    for img in supportList:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
        hsv = hsv + 10
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        positiveSample.append(img)
            
        hsv = hsv - 20
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        positiveSample.append(img)

    return


def showImages(images):
    """
    Helper function to view images generated in the script without having to store
    them on the disk. Use 'a' and 'd' key to go to the next image.
    """
    idx = 0

    while True:

        cv2.imshow("Image", images[idx])

        if cv2.waitKey(15) & 0xFF == ord("d"):
            if idx+1 >= len(images):
                print("This is the last image in the set.")
            else:
                idx += 1
                print("Viewing image no. {0} / {1}".format(idx+1, len(images)))

        if cv2.waitKey(15) & 0xFF == ord("a"):
            if idx-1 < 0:
                print("This is the first image in the set.")
            else:
                idx -= 1
                print("Viewing image no. {0} / {1}".format(idx+1, len(images)))

        if cv2.waitKey(15) & 0xFF == ord("q"):
            break

def computeHOG(images, hogList, size=(64, 64)):
    """
    This function computes the Histogram of Oriented Gradients (HOG) of each
    image from the dataset.
    [Code from Exercise 10 solution. Could be used for a SVM with HOG Features]
    """
    # Creates a HOG descriptor with custom parameters
    # (only changed the window size from the default settings to function
    # correctly for our 64x64 input images)
    # see (https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
    hog = cv2.HOGDescriptor("./inputs/hog.xml")
    
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

    return 

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

    # As 3.02. (a) : Load our car images dataset.
    positiveList = loadDataset(positiveFile)
    negativeList = loadDataset(negativeFile)


    print("Initial size of car set: {0} \t\t (dim: {1})".format(len(positiveList), positiveList[0].shape))
    print("Initial size of non-car set: {0} \t\t (dim: {1})".format(len(negativeList), negativeList[0].shape))


    # As 3.02. (b) : Get a sample of negative images. (returns list in negativeSample)
    sampleNegativeImages(negativeList, negativeSample, size=(64,64), N=200)
    
    # As 3.02. (c) : [EXTRA] increase the car dataset by generating new positive images
    samplePositiveImages(positiveList, positiveSample, size=(64,64), N=200)
    

    print("Size of non-car sample set: {0} \t (dim: {1})".format(len(negativeSample), negativeSample[0].shape))
    print("Size of car sample set: {0} \t\t (dim: {1})".format(len(positiveSample), positiveSample[0].shape))

    #--------------------------------------------------#
    #                                                  #
    # Classification Model using SVM with HOG Features #
    #                                                  #
    #--------------------------------------------------#

    # Computing the HOG features for each image
    hogList = []
    computeHOG(positiveSample, hogList, size=(64,64))
    computeHOG(negativeSample, hogList, size=(64,64))
    hogList = [vec.flatten() for vec in hogList]

    print("Vector Length of one HOG: {}".format(len(hogList[0])))

    # create the labels (1: car, -1: non-car)
    [labels.append(+1) for _ in range(len(positiveSample))]
    [labels.append(-1) for _ in range(len(negativeSample))]

    # Split into a train/test/validation set (70/15/15)
    np_labels = np.array(labels).reshape(len(labels),1)
    np_hogs = np.array(hogList)
    dataset = np.hstack((np_hogs,np_labels))
    
    np.save('./outputs/dataset.npy', dataset)

    # store the 2500 images in a separate output folder
    if not os.path.isdir("./outputs/extra_images/"):
        os.makedirs("./outputs/extra_images/")

    file_names = []
    idx = 0
    for image in (positiveSample + negativeSample):
        fname = "./outputs/extra_images/Cars_" + str(idx) + "_Extra.png"
        cv2.imwrite(fname, image)
        file_names.append(fname) 
        idx += 1
    print("Done storing the " + str(len(positiveSample+negativeSample)) + " images.")

    # also store as CSV 
    df = pd.DataFrame(data={
                    'files' : file_names,
                    'HOG'   : [row for row in dataset[:,:-1]], 
                    'label' : dataset[:,-1]})
    df.to_csv("./outputs/hog_dataset.csv")

    X_train, X_test, y_train, y_test = train_test_split(dataset[:,:-1], dataset[:,-1], test_size=0.15, random_state=1)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.1765, random_state=1)

    print("sizes of train/validation/test sets: {0}/{1}/{2}".format(X_train.shape[0],X_val.shape[0],X_test.shape[0]))

    kernel = "rbf" # "linear"

    svc = svm.SVC(kernel=kernel, probability=True, class_weight='balanced')
    svc.fit(X_train,y_train)
    
    # store prediction results on the validation set
    train_pred  = svc.predict(X_train)
    val_pred    = svc.predict(X_val)
    test_pred   = svc.predict(X_test)

    train_acc    = sklearn.metrics.accuracy_score(y_train,train_pred)
    val_acc      = sklearn.metrics.accuracy_score(y_val, val_pred)
    test_acc     = sklearn.metrics.accuracy_score(y_test, test_pred)
    print("Accuracy on the training set: \t\t {number:.{digit}f}".format(number=train_acc, digit=3))
    print("Accuracy on the validation set: \t {number:.{digit}f}".format(number=val_acc, digit=3))

    # confusion matrix on the validation set
    print("Confusion on the validation set.")
    print("1st Col/Row: Non-Cars | 2nd Col/Row: Cars")
    print(sklearn.metrics.confusion_matrix(y_val, val_pred))

    print("\n\nAccuracy on the test set: \t {number:.{digit}f}".format(number=test_acc, digit=3))
    print("Confusion on the test set.")
    print("1st Col/Row: Non-Cars | 2nd Col/Row: Cars")
    print(sklearn.metrics.confusion_matrix(y_test, test_pred))

    joblib.dump(svc, './inputs/svm_model_weights_' + kernel + '.pkl') 

#<!--------------------------------------------------------------------------->
#<!--                                                                       -->
#<!--------------------------------------------------------------------------->

# put executing code in main, so that the defined functions
# can be imported in a separate script without executing
# the code
if __name__ == "__main__":
    main()

