# importing the necessary libraries
import cv2
import numpy as np
from time import sleep, time
from numba import jit

# Creating a VideoCapture object to read the video
# cap = cv2.VideoCapture('contrast_1280x720.mp4')
cap = cv2.VideoCapture("contrast_1920x1080.mp4")

# optimized version
@jit(nopython=True)
def diffuse_across_edges_optimized(img, epsilon=100, b=0.1):
    gamma = 1e-5
    padded = np.zeros((original.shape[0] + 2, original.shape[1] + 2, 3)).astype("uint8")
    padded[1:-1, 1:-1] = img
    img_new = padded.copy()
    max_diff = epsilon + 1
    while max_diff > epsilon:
        central = padded[1:-1, 1:-1]
        laplacian = (
            padded[1:-1, 2:]
            + padded[1:-1, :-2]
            + padded[2:, 1:-1]
            + padded[:-2, 1:-1]
            - 4 * central
        )
        # use square difference to quadratically penalize very large laplacians
        diff = gamma * np.absolute(np.power(laplacian, 2) - epsilon)
        img_new[1:-1, 1:-1] = central + b * diff * laplacian
        padded = img_new
        max_diff = np.max(diff)
        print(max_diff)
    return padded[1:-1, 1:-1]


@jit(nopython=True)
def diffuse_across_edges(img, epsilon=0.3, b=2e3):
    padded = np.zeros((original.shape[0] + 2, original.shape[1] + 2, 3)).astype("uint8")
    padded[1:-1, 1:-1] = img
    img_new = padded.copy()
    central = padded[1:-1, 1:-1].astype("int64")
    laplacian = (
        padded[1:-1, 2:]
        + padded[1:-1, :-2]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        - 4 * central
    )
    laplacian = (laplacian / 1e3)
    max_diff = np.max(np.absolute(laplacian))
    while max_diff > epsilon:
        img_new[1:-1, 1:-1] = (central + b * np.power(laplacian, 3))
        padded = img_new        

        central = padded[1:-1, 1:-1].astype("int64")
        laplacian = (
            padded[1:-1, 2:]
            + padded[1:-1, :-2]
            + padded[2:, 1:-1]
            + padded[:-2, 1:-1]
            - 4 * central
        )
        laplacian = (laplacian / 1e3)

        entropy = np.sum(np.power(laplacian, 2))
        max_diff = np.max(np.absolute(laplacian))
    return padded[1:-1, 1:-1]




# arm
# w = 200
# h = 200
# left = 700
# top = 500
# offset = 1950
# grid_size = 400

# upper body
w = 500
h = 500
left = 600
top = 200
offset = 1920
grid_size = 500


output_size = (3 * grid_size, grid_size)

out = cv2.VideoWriter(
    "output.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 30, output_size
)

def gradient_square_entropy(img):
    gradient = cv2.Sobel(img, cv2.CV_16S, 1, 1)
    gradient = gradient / 1e3
    return np.round(np.sum(np.power(gradient, 2)), 2)


while cap.isOpened():
    t = time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break

    height, width = frame.shape[:2]

    original = frame[top : top + h, left : left + w]
    hardware_antialiased = frame[top : top + h, left + offset : left + offset + w]

    diffused = diffuse_across_edges(original, 0.4, 1e3)
    gaussian = cv2.GaussianBlur(original, (3,3), 100)

    frame = np.concatenate([diffused, gaussian, original], axis=1)
    frame = cv2.resize(frame, output_size)


    diffused_entropy = gradient_square_entropy(diffused)
    original_entropy = gradient_square_entropy(original)
    gaussian_entropy = gradient_square_entropy(gaussian)
    hardware_entropy = gradient_square_entropy(hardware_antialiased)

    frame = cv2.putText(
        frame,
        f"diffused: relative entropy {round(diffused_entropy - original_entropy)}",
        (10, 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (255, 255, 255),
        1,
    )
    frame = cv2.putText(
        frame,
        f"gaussian: relative entropy {round(gaussian_entropy - original_entropy)}",
        (grid_size + 10, 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (255, 255, 255),
        1,
    )
    frame = cv2.putText(
        frame,
        f"original: relative entropy {0}",
        (grid_size * 2 + 10, 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (255, 255, 255),
        1,
    )
    
    
    # Display the resulting frame
    cv2.imshow("Frame", frame)
    out.write(frame)

    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    print(f"running at {1/(time() - t)} hz")
# release the video capture object
cap.release()
out.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()
