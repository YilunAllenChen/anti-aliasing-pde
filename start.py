# importing the necessary libraries
import cv2
import numpy as np
from time import sleep
 
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('source.mp4')
 
 
# Loop until the end of the video
while (cap.isOpened()):
 
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)
 
    # Display the resulting frame
    cv2.imshow('Frame', frame)
 
    # conversion of BGR to grayscale is necessary to apply this operation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # adaptive thresholding to use different threshold
    # values on different regions of the frame.
    Thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
 
    cv2.imshow('Thresh', Thresh)
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    sleep(0.01)
    
 
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()