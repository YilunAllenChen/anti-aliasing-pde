# importing the necessary libraries
import cv2
import numpy as np
from time import sleep, time
from numba import jit
 
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('contrast_1280x720.mp4')

# @jit(nopython=True)
# def linear_heat(img, steps=100, gamma = 0.01):
#     w, h = img.shape[:2]
#     for t in range(steps):
#         img_new = img.copy()
#         for i in range(1, w - 1):
#             for j in range(1, h - 1):
#                 diff = gamma * (img[i-1, j] + img[i+1, j] + img[i, j+1] + img[i, j-1] - 4*img[i,j])
#                 if sum(np.abs(diff)) > 0.3:
#                     img_new[i, j] = img[i, j] + diff
#         img = img_new.copy()
#     return img

# optimized version
@jit(nopython=True)
def linear_heat(img, steps=100, gamma=0.01):
    img_new = img.copy()
    for t in range(steps):
        img_new[1:-1, 1:-1] = (1 - 4 * gamma) * img[1:-1, 1:-1] + gamma * (img[1:-1, 2:] + img[1:-1, :-2] + img[2:, 1:-1] + img[:-2, 1:-1])
        img = img_new
    return img[1:-1, 1:-1]
# Loop until the end of the video



while (cap.isOpened()):
    t = time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    # frame = cv2.resize(frame, (640, 360), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)

    height, width = frame.shape[:2]
    w = 600
    h = 600
    left = 500
    top = 200
    offset = 1920
    
    hardware_antialiased = frame[top:top+h, left:left+w]
    original = frame[top:top+h, left+offset:left+offset+w]

  
    # original[original==0] = 1
    

    edges = cv2.Sobel(original, cv2.CV_8U, 1, 1, ksize=3)
    edges = cv2.dilate(edges, np.ones((3,3)))
    mask = edges > 80
    
    
    padded = np.zeros((original.shape[0]+2, original.shape[1]+2,3)).astype('uint8')

    padded[1:-1, 1:-1] = original 
    heat_res = original.copy().squeeze()
    diffused = linear_heat(padded, 3, 0.15)
    heat_res[mask] = diffused.squeeze()[mask]


    
    
    

    
    frame = np.concatenate([heat_res, hardware_antialiased, edges], axis=1)

    # frame = cv2.resize(frame, (w * 4, h * 2))
 
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    
 
    # conversion of BGR to grayscale is necessary to apply this operation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    print(f"running at {1/(time() - t)} hz")
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()