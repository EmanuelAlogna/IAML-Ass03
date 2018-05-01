#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : warmUpAnnotations.py                                     -->
#<!-- Description: Example of code for select regions of interest in images -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : narcizo[at]itu[dot]dk                                    -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 24/04/2018                                               -->
#<!-- Change     : 24/04/2018 - Creation of this script                     -->
#<!-- Review     : 24/04/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018042401 $"

###############################################################################
import cv2
import time
import os

###############################################################################
# Global variables.
points = []
cropping = False

def saveImage(image):
    """Save a new positive image in the dataset."""
    # Folder where the image will be saved.
    folder = "./inputs/cars/"

    # Current image ID.
    index = 0
    while True:
        # Create the image filename.
        filename = "%sCars_%03d.png" % (folder, index + 1)

        # Check if it is available in the folder and avoid to overwrite it.
        if not os.path.isfile(filename):

            # Save the image as grayscale.
            cv2.imwrite(filename, image)
            print("Saved: %s" % filename)            
            break

        # Try to use a new image ID.
        index += 1

def selectROI(event, x, y, flags, param):
    """Select a region of interest using the mouse."""
    # Global variables.
    global points, cropping

    # Event for the left mouse button click.
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        cropping = True

    # Event for the left mouse button release.
    elif event == cv2.EVENT_LBUTTONUP:
        # Check the upper left corner.
        if len (points) == 0:
            return

        # Calculate the width and height of bouding box.
        width  = int(x) - points[0][0]
        height = int(y) - points[0][1]

        # Aspect ratio.
        if width < height:
            width  = int(height)
        else:
            height = int(width)

        # Record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished.
        points.append((points[0][0] + width, points[0][1] + height))
        cropping = False

        # Draw a rectangle around the region of interest.
        image = frame.copy()
        cv2.rectangle(image, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow("Video", image)

        # Press the key "S" on your keyboard to save the selected ROI.
        key = cv2.waitKey(0)
        if key == ord("s"):
            roi = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]
            roi = cv2.resize(roi, (64, 64))            
            saveImage(roi)


# Defines the filepath.
filepath = "./inputs/videos/Cars_05.mov"

# Create a capture video object.
capture = cv2.VideoCapture(filepath)

# Get the video frame rate.
fps = int(round(capture.get(cv2.CAP_PROP_FPS)))

# Check if the fps variable has a correct value.
fps = fps if fps > 0 else 30

# Create an OpenCV window.
cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Video", selectROI)

# Press the key "P" on your keyboard to pause the video.
isPaused = False

# This repetion will run while there is a new frame in the video file or
# while the user do not press the "q" (quit) keyboard button.
while True:
    # Capture frame-by-frame.
    if not isPaused:
        retval, frame = capture.read()

    # Check if there is a valid frame.
    if not retval:
        break

    # Display the resulting frame.
    cv2.imshow("Video", frame)

    # Check the keyboard events.
    key = cv2.waitKey(fps)
    if key == ord("q"):
        break
    elif key == 32:
        isPaused = not isPaused 

# When everything done, release the capture and record objects.
capture.release()
cv2.destroyAllWindows()
