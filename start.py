# importing the necessary libraries
import cv2
import numpy as np
from time import sleep, time
from numba import jit
 
# Creating a VideoCapture object to read the video
# cap = cv2.VideoCapture('contrast_1280x720.mp4')
cap = cv2.VideoCapture('contrast_1920x1080.mp4')


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
    padded = np.zeros((original.shape[0]+2, original.shape[1]+2,3)).astype('uint8')
    padded[1:-1, 1:-1] = img 
    img_new = padded.copy()
    for t in range(steps):
        img_new[1:-1, 1:-1] = (1 - gamma * 4) * padded[1:-1, 1:-1] + gamma * (padded[1:-1, 2:] + padded[1:-1, :-2] + padded[2:, 1:-1] + padded[:-2, 1:-1])
        padded = img_new
    return padded[1:-1, 1:-1]

# optimized version
@jit(nopython=True)
def linear_heat_epsilon(img, epsilon=100, gamma=0.1):
    padded = np.zeros((original.shape[0]+2, original.shape[1]+2,3)).astype('uint8')
    padded[1:-1, 1:-1] = img 
    img_new = padded.copy()
    max_diff = epsilon + 1
    while(max_diff > epsilon):
        central = padded[1:-1, 1:-1]
        diff = gamma * (padded[1:-1, 2:] + padded[1:-1, :-2] + padded[2:, 1:-1] + padded[:-2, 1:-1] - 4 * central)
        diff = diff * (np.power(diff, 2) > epsilon)
        img_new[1:-1, 1:-1] = central + diff
        padded = img_new
        max_diff = np.max(diff)
    return padded[1:-1, 1:-1]


w = 600
h = 600
left = 500
top = 200
offset = 1920
 
# dilation_kernal = np.array([
#     [0, 1, 2, 1, 0],
#     [1, 2, 4, 2, 1],
#     [2, 4, 8, 4, 0],
#     [1, 2, 4, 2, 1],
#     [0, 1, 2, 1, 0]
# ]).astype("uint8")

dilation_kernal = np.array([
    [0, 1, 0],
    [1, 2, 1],
    [0, 1, 0]
]).astype('uint8')

while (cap.isOpened()):
    t = time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break

    height, width = frame.shape[:2]
   
    original = frame[top:top+h, left:left+w]
    hardware_antialiased = frame[top:top+h, left+offset:left+offset+w]
  

    laplacian = cv2.Laplacian(original, cv2.CV_8U)
    edges = np.absolute(cv2.Sobel(original, cv2.CV_8U, 1, 1, ksize=3))


    dilated = cv2.dilate(laplacian, dilation_kernal)
    mask = dilated > 100
    
    # diffused = linear_heat(original, 2, 0.15)
    diffused = linear_heat_epsilon(original, 60, 0.2)
    

    heat_res = original.copy().squeeze()
    heat_res[mask] = diffused.squeeze()[mask]
    
    frame = np.concatenate([heat_res, original, hardware_antialiased], axis=1)

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