# importing the necessary libraries
import cv2
import numpy as np
from time import sleep, time
from numba import jit
from matplotlib import pyplot as plt

# Creating a VideoCapture object to read the video

# low resolution version
# cap = cv2.VideoCapture('contrast_1280x720.mp4')

# high resolution version
cap = cv2.VideoCapture("contrast_1920x1080.mp4")


def inverse_perona_malik_diffusion(img, epsilon=0.2, b=0.03):
    copy = img.copy()
    gradient = cv2.Sobel(copy, cv2.CV_16S, 1, 0) + cv2.Sobel(copy, cv2.CV_16S, 0, 1)
    max_grad = np.max(gradient / 1e3)
    dx = 1000
    print(max_grad)
    while max_grad > epsilon:
        Ix = cv2.Sobel(copy, cv2.CV_32F, 1, 0) / dx
        Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0) / dx
        Iy = cv2.Sobel(copy, cv2.CV_32F, 0, 1) / dx
        Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1) / dx
        It = b * dx * ((Ix**2 + Iy**2)**2) * (Ixx + Iyy)
        diffused = (copy + It)
        copy = diffused.astype("uint8")
        gradient = cv2.Sobel(copy, cv2.CV_16S, 1, 0) + cv2.Sobel(copy, cv2.CV_16S, 0, 1)
        max_grad = np.max(gradient / 1e3)
        print("gradient: ", max_grad)
    return copy


# arm
w = 200
h = 200
left = 700
top = 500
offset = 1950
grid_size = 400

# upper body
# w = 500
# h = 500
# left = 700
# top = 200
# offset = 1920
# grid_size = 500


output_size = (2 * grid_size, 2 * grid_size)
out = cv2.VideoWriter(
    "output.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 30, output_size
)


def gradient_square_entropy(img):
    gradient = cv2.Sobel(img, cv2.CV_16S, 1, 1)
    gradient = gradient / 1e3
    return np.round(np.sum(np.power(gradient, 2)), 2)

entropies = {
    "diffused": [],
    "gaussian": [],
    "original": [],
    "hardware": []
}

while cap.isOpened():
    t = time()
    ret, frame = cap.read()
    # Capture frame-by-frame
    if frame is None:
        break

    height, width = frame.shape[:2]

    # extract the original graphics and the FXAA-based antialiasing graphics
    original = frame[top : top + h, left : left + w]
    hardware_antialiased = frame[top : top + h, left + offset : left + offset + w]
   
    # For comparison purposes, include a Gaussian 
    gaussian = cv2.GaussianBlur(original, (3,3), 100)

    # Use the inverse perona malik diffusion
    diffused = inverse_perona_malik_diffusion(original, 1, 50)

    #
    frame = np.concatenate(
        (
            np.concatenate([diffused, original], axis=1), 
            np.concatenate([gaussian, hardware_antialiased], axis=1)
        ),
        axis=0)
    frame = cv2.resize(frame, output_size)


    diffused_entropy = gradient_square_entropy(diffused)
    original_entropy = gradient_square_entropy(original)
    gaussian_entropy = gradient_square_entropy(gaussian)
    hardware_entropy = gradient_square_entropy(hardware_antialiased)

    entropies['diffused'].append(diffused_entropy)
    entropies['original'].append(original_entropy)
    entropies['gaussian'].append(gaussian_entropy)
    entropies['hardware'].append(hardware_entropy)

    font_size = 0.8
    frame = cv2.putText(
        frame,
        f"diffused: relative entropy {round(diffused_entropy - original_entropy)}",
        (10, 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        font_size,
        (255, 255, 255),
        1,
    )
    frame = cv2.putText(
        frame,
        f"original: relative entropy {0}",
        (grid_size + 10, 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        font_size,
        (255, 255, 255),
        1,
    )
    frame = cv2.putText(
        frame,
        f"Gaussian: relative entropy {round(gaussian_entropy - original_entropy)}",
        (10, grid_size + 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        font_size,
        (255, 255, 255),
        1,
    )
    frame = cv2.putText(
        frame,
        f"FXAA: relative entropy {round(hardware_entropy - original_entropy)}",
        (grid_size + 10, grid_size + 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        font_size,
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

print(entropies)
ground_truth_entropy = entropies['hardware']
t = [i for i in range(len(entropies['diffused']))]
for (key, val) in entropies.items():
    val = np.abs(np.array(val) - np.array(ground_truth_entropy))
    plt.plot(t, val, label=key)
plt.legend()
plt.show()