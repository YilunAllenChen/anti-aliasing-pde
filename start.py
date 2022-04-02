# importing the necessary libraries
import cv2
import numpy as np
from time import sleep, time
from numba import jit
 
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('contrast_1280x720.mp4')
# cap = cv2.VideoCapture('contrast_1920x1080.mp4')


# legacy code that runs very slowly.
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

w = 600
h = 600
left = 500
top = 200
offset = 1920
 


while (cap.isOpened()):
    t = time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break

    height, width = frame.shape[:2]
   
    original = frame[top:top+h, left:left+w]
    hardware_antialiased = frame[top:top+h, left+offset:left+offset+w]

  

    edges = cv2.Sobel(original, cv2.CV_8U, 1, 1, ksize=3)
    edges = cv2.dilate(edges, np.ones((5,5)))
    mask = edges > 100 
    
    
    padded = np.zeros((original.shape[0]+2, original.shape[1]+2,3)).astype('uint8')
    padded[1:-1, 1:-1] = original 
    diffused = linear_heat(padded, 2, 0.1)

    non_discriminating = diffused.copy()

    heat_res = original.copy().squeeze()
    heat_res[mask] = diffused.squeeze()[mask]

    frame = np.concatenate([heat_res, non_discriminating, original], axis=1)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
 
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    print(f"running at {1/(time() - t)} hz")
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()