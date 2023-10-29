# PDE-Based Anti-Aliasing in Computer Graphics
Apr. 5th, 2022

# Part I: Description of Problem

Anti-aliasing is a technique widely used in computer graphics (especially computer games) to better the graphical appearance of objects by removing the “jaggies” that are typically seen along edges or boundaries of these objects. Modern anti-aliasing algorithms like FXAA, TSAA and SMAA leverage super-sampling (both spatial and temporal) to obtain multiple samples within the same pixel, averaging them out to render boundary pixels as transitioning pixels between different objects, smoothing the edges. 

![**Figure:** Jaggieness in computer graphics](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/Untitled.png)

**Figure:** Jaggieness in computer graphics

However, these techniques put significant computational load on the graphics card (by requiring sampling and hence computing what to render on a pixel multiple times for a single frame), or takes up a considerable amount of graphical memory (by storing pixel values of previous frames). In this short project, we aim to explore a way to perform anti-aliasing that requires much less samples and takes up much less memory, by leveraging PDE techniques.

![**Figure 2**: Comparison of the same graphics with Anti-aliasing off (Left) and on (right)](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/Untitled%201.png)

**Figure 2**: Comparison of the same graphics with Anti-aliasing off (Left) and on (right)

Please note that computational efficiency is not taken into consideration in this project, because in order for the technique to be truly feasible/comparable with traditional techniques, this computation must happen on a bare-metal hardware level, instead of on software level.

# Part II: Problem Formulation

To achieve the goal of removing jaggieness in the graphics, the jaggieness must first be located. These features can be described mathematically to be having a large gradient (and hence large values in its Laplacian in the vicinity) - the value of a pixel is drastically different from its neighboring pixels. 

To obtain the gradient and the Laplacian in the image, a discretization, as well as the corresponding CFL condition must be derived. 

These pixels, once identified, will be processed using a special form of the heat equations with a pre-determined diffusion constant and pre-determined threshold value. This allows the pixel values to diffuse, “blending” with its neighboring pixels, smoothing out the overall change in pixel values.

However, in order to formulate this “discrimination” mathematically, one can assign large weights to pixels where Laplacian is large, promoting diffusion in these areas and demoting diffusion elsewhere, instead of completely turning it off (doing that will also introduce discontinuity).

In the meantime, one must take into practical consideration that “jaggieness” refers to the stair-case-like appearance of pixels to our eyes, not the rendering process or the pixel values themselves. This ideation allows one to exclude from processing, the pixels that have sharp edges but don’t appear to be jaggy to our eyes. These pixels reside on the edges of objects that have similar color with its neighbors. It is also worth noting that if one does heat diffusion in a non-selective manner for all of the pixels (just like a global Gaussian blur), then some of the features in the original graphics will be lost. Or to put it in other words, graphics will look blurry, not as sharp as before. Therefore a balance point must be found.

To address all these concerns, a special across-edge heat diffusion inspired by the Perona-Malik equation will be used.

For implementation, gradient descent methods will be used to converge to approximate solution instead of deriving and solving the equations analytically. This is acceptable because, again, the resolution of the output values is low, hence the extra precision is not needed. 

# Part III: Deriving The PDE’s

Note the Perona-Malik with a linear $c$:

$$
\begin{equation}
I_t=\frac{1}{||\nabla I||}\frac{I_x^2I_{yy}-2I_xI_yI_{xy}+I_y^2I_{xx}}{(I_x^2+I_y^2)}
\end{equation}
$$

The $\frac{1}{||\nabla I||}$term controls the direction of diffusion, promoting diffusion along edges (the bigger the gradient, the slower the diffusion), not across. However, this edge-preserving-noise-cancelling behavior is exactly the opposite of what is required in this problem formulation. Since computer graphics are rendered (and hence noise free), performing “noise-cancelling” will only blur existing features, changing the texture and appearance of objects. Moreover, we want diffusion to happen across edges so that the jaggieness can be smoothed out. An intuitive way to think about is, we want to have a formulation that emulate the form: 

$$
\begin{equation}
I_t=||\nabla I||\frac{I_x^2I_{yy}-2I_xI_yI_{xy}+I_y^2I_{xx}}{(I_x^2+I_y^2)}
\end{equation}
$$

such that areas where gradients (and hence the magnitude of Laplacians) are large will promote the diffusion process. For simplicity purposes, however, linear heat diffusion will be used in the isotropic part of the formulation in this experiment, instead of geometric heat. Generalizations and other potential improvements will be discussed in Part VI.

$$
\begin{equation}
I_t=||\nabla I||b\Delta I
\end{equation}
$$

Moreover, to further penalize large gradient areas, the gradient term is raised to the 4th power such that jaggieness are better captured in the diffusion.

$$
\begin{equation}
I_t=||\nabla I||^4b\Delta I=(I_x^2+I_y^2)^2b(I_{xx}+I_{yy})
\end{equation}
$$

Where:

- $b$: heat diffusion constant. It impacts how fast the heat diffusion converges. However, the bigger it is, the bigger the step size, hence the less precision one retain from gradient descent (and along with it, the risk of overshoot). Later it will also be proven that it cannot exceed a CFL-condition-set upper bound, otherwise it will de-stabilize the system.

Note that in this time evolution, the steady state where all edges are smoothed out is not desired (that would translate to a plain-color image that bears the average pixel values of the original picture). Therefore, an additional variable will be introduced to mark the end condition of the gradient descent. The significance of this threshold variable and how to choose it will be discussed in Part VI.

- $\epsilon$: The threshold value that signifies the end of the calculation when all Laplacian values fall below this threshold. In other words, equation (4) is amended to become:

$$
\begin{equation}
I_t=
\begin{cases}
(I_x^2+I_y^2)^2b(I_{xx}+I_{yy}), & \max|\nabla I|>\epsilon \\
0, &\max|\nabla I| < \epsilon \\
\end{cases}
\end{equation}
$$

# Part IV: Discretization and Implementation of PDE’s

To obtain the Laplacian, the central different is used. First order central difference is used instead of other formulations because of the following reasons:

- Higher order differences: These methods requires samples that are further away, and hence requires taking more samples from the original pictures, increasing the computation cost. Since the color output is merely `uint8` (hence 0 ~ 255) and the amount of extra precision one can extract from using these terms is very limited, the payout for such computation is disproportional to the extra workload introduced.
- Backward / Forward differences: No significant advantages / drawbacks. One trivial drawback is that because the original image would have been padded asymmetrically, it might cause some extra diffusion to take place on two of the four edges, instead of evenly on all four edges.
- Implicit formulation: These approaches are most useful when one doesn’t have access to spatial neighbors of the pixel interested at the current timestamp, but that’s not the case in this problem. This additional “flexibility” typically requires one to solve a linear system, introducing extra computational cost while offering no obvious advantages (since in this problem one has the resources that implicit formulation assumes non-existent).

Determining the stability and the CFL condition: 

First, note the original formulation:

$$
\begin{equation}I_t=||\nabla I||^4 (\Delta I)=(I_x^2+I_y^2)^2(I_{xx}+I{yy})\end{equation}
$$

Is a non-linear PDE, but is quasi-linear. We therefore attempt to linearize the equation in time, hence transforming the equation into one that behaves like a linear heat equation:

$$
\begin{equation}\mathfrak{B}(I_x,I_y)= b(I_x^2+I_y^2)^2\end{equation}
$$

which yields the diffusion equation:

$$
\begin{equation}I_t=\mathfrak{B}(I_x, I_y) \Delta I\end{equation}
$$

Note that since the diffusion will cause the gradient of $I$ to strictly decrease, we have:

$$
I_t(x,y,t)\le I_t(x,y,t-\epsilon)\forall x,y,t>0, t-\epsilon>0
$$

Therefore it is safe to claim:

$$
\max_{x,y}I_t(x,y,t)\le \max_{x,y}I_t(x,y,t-\epsilon), \forall t>0,t-\epsilon>0
$$

It can be concluded that $\mathfrak{B}(I_x, I_y)$ is strictly decreasing with respect to time. Therefore a conservative approximation can be made by setting it to its maximum possible value

$$
\begin{equation}I_t\le\max_{I_x, I_y}(\mathfrak{B}(I_x, I_y)) \Delta I=\mathfrak{B}_{max}\Delta I\end{equation}
$$

We can hence discretize it (approximating the Laplacian with central difference): 

$$
\begin{equation}u(x,y,t+\Delta t)=u(x,y,t)+\Delta t\mathfrak{B}_{max}(\frac{u(x+\Delta x,y,t)+u(x-\Delta x,y,t)-2u(x,y,t)}{\Delta x^2}+\frac{u(x,y+\Delta y,t)+u(x,y-\Delta y,t)-2u(x,y,t)}{\Delta y^2})\end{equation}
$$

Since vertical and horizontal pixels are of the same size, hence using the relationship $\Delta x=\Delta y:$

$$
\begin{equation}u(x,y,t+\Delta t)=u(x,y,t)+\frac{\Delta t\mathfrak{B}_{max}}{\Delta x^2}({u(x+\Delta x,y,t)+u(x-\Delta x,y,t)-2u(x,y,t)}+{u(x,y+\Delta y,t)+u(x,y-\Delta y,t)-2u(x,y,t)}))\end{equation}
$$

Combining terms: 

$$
\begin{equation}u(x,y,t+\Delta t)=u(x,y,t)+\frac{\Delta t\mathfrak{B}_{max}}{\Delta x^2}({u(x+\Delta x,y,t)+u(x-\Delta x,y,t)}+{u(x,y+\Delta y,t)+u(x,y-\Delta y,t)-4u(x,y,t)}))\end{equation}
$$

Using the Fourier Transform:

$$
\begin{equation}U(\omega,\omega,t+\Delta t)=U(\omega,\omega,t)+\frac{\Delta t\mathfrak{B}_{max}}{\Delta x^2}(2e^{j\Delta x\omega}+2e^{-j\Delta x\omega}-4)U(\omega,\omega,t)\end{equation}
$$

By reorganizing terms, we obtain the scaling factor $\alpha$:

$$
\begin{equation}\alpha(\omega)=\frac{U(\omega, \omega,t+\Delta t)}{U(\omega, \omega, t)}=1+\frac{\Delta t\mathfrak{B}_{max}}{\Delta x^2}(2e^{j\Delta x\omega}+2e^{-j\Delta x\omega}-4)
\end{equation}
$$

With stability condition stated as:

$$
\begin{equation}
|\alpha|\le 1
\end{equation}

$$

Note that using Euler’s formula, we can obtain from equation (14): 

$$
\begin{equation}
\alpha(\omega)=1+\frac{4\Delta t\mathfrak{B}_{max}}{\Delta x^2}(\cos(\omega\Delta x) - 1)
\end{equation}

$$

Hence we have:

$$
\begin{equation}
\frac{4\Delta t\mathfrak{B}_{max}}{\Delta x^2}(\cos(\Delta x\omega)-1) < 2
\end{equation}

$$

Where: 

$$
\begin{equation}
\max_{\Delta x, \omega}(\cos(\Delta x\omega)-1)=2
\end{equation}
$$

hence to enforce (15), we reach the conservative CFL condition:

$$
\begin{equation}
\frac{4\Delta t\mathfrak{B}_{max}}{\Delta x^2}\le1 \rightarrow \Delta t\le\frac{\Delta x^2}{4\mathfrak{B}_{max}}

\end{equation}
$$

Since in this problem it is not of interest what particular values $\Delta x$ and $\Delta t$ take in, they can be chosen arbitrarily as long as they comply with the CFL condition.

Note that the gradient descent defined earlier, once discretized, can be viewed as:

$$
\begin{equation}
u(x,y,t+\Delta t)=
\begin{cases}
u(x,y,t)+\frac{4\Delta tb}{\Delta x^2}(||\nabla u(x,y,t)||^2\Delta u(x,y,t)), & \max(\nabla u(x,y,t))>\epsilon \\
u(x,y,t), & \max(\nabla u(x,y,t))<\epsilon
\end{cases}
\end{equation}
$$

Therefore it can be implemented as follows (pseudocode): 

```python
def inverse_perona_malik_diffusion(img, epsilon=0.2, b=0.03):
    max_grad = max(gradient(copy))
    while max_grad > epsilon:
        It = b * (Ix**2 + Iy**2)**2 * (Ixx + Iyy)
        diffused = (copy + It)
        copy = diffused.astype("uint8")
        max_grad = max(gradient(copy))
    return copy
```

### Full Code is here:

 [https://github.com/YilunAllenChen/ECE6560](https://github.com/YilunAllenChen/ECE6560)

A copy of the main script is attached in Appendix A.

# Part V: Experimental Results

As a metric of the experiment, an entropy function is introduced that penalizes large derivatives values:

$$
\begin{equation}
H(I, I_x, I_y)=\frac{1}{z}\int(\nabla I)^2dxdy
\end{equation}
$$

Or to put it in discrete form:

$$
H(u, \nabla u) = \frac{1}{z}\sum_{x,y}\nabla u(x,y)
$$

Where $\nabla u$ will be calculated using the Sobel filter, and $z$ is a normalizing constant that is static across different measurements

The entropy for the image after diffusion will be compared with the original picture as well as the one processed using traditional anti-aliasing techniques. Once subtracted the entropy of the original picture, a negative value would mean that the overall entropy went down, hence the picture is smoothed out.

Below are some example results (video source: StarCraft II)

![**Figure 3:** Comparison between proposed diffusion (left), original picture (middle) and FXAA-based anti-aliasing (right).](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/Untitled%202.png)

**Figure 3:** Comparison between proposed diffusion (left), original picture (middle) and FXAA-based anti-aliasing (right).

It is worth noting that the appearance of the character’s face has changed slightly (face texture slightly smoothed out, and the highlight in the eyes appear dimmer). This compromise is unavoidable due to the nature of the algorithm. However, it does provide an edge over completely non-discriminating diffusion (like Gaussian blurs), which will be discussed in Part VI.

![**Figure 3:** Enlarged picture around the arms. Proposed diffusion (left), original picture (middle) and FXAA-based anti-aliasing (right).](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/Untitled%203.png)

**Figure 3:** Enlarged picture around the arms. Proposed diffusion (left), original picture (middle) and FXAA-based anti-aliasing (right).

It is clear that the proposed diffusion smooths out jaggieness in the pictures (near the edge of arms) while mostly preserving the textures and appearance of objects (the dotted dark-blue texture of the cloth on the top right of the pictures).

To better demonstrate the advantage of preserving textures and appearance of objects, a comparison between the proposed diffusion and linear heat diffusion (Gaussian blur with kernal size 3x3) is drawn:

![**Figure 5:** Comparison amongst the proposed diffusion (left), Gaussian Blur (middle) and original picture(right).](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/Untitled%204.png)

**Figure 5:** Comparison amongst the proposed diffusion (left), Gaussian Blur (middle) and original picture(right).

It is obvious that, albeit Gaussian blur achieves a lower overall entropy and that edges are indeed smoothed out, some of the desired sharpness that contribute to the overall quality of the graphics is lost (for example, the white circuitry on the dark-blue cloth near character’s left shoulder). The proposed diffusion, in contrast, selectively smooths out edges by heavily penalizing large gradients. 

![**Figure 6:** Enlarged picture near the character’s arm.](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/Untitled%205.png)

**Figure 6:** Enlarged picture near the character’s arm.

A similar conclusion can be drawn from the comparison above. Although Gaussian blur achieves an overall lower entropy, the texture of the cloth (dotted-ness) is lost during the blurring process. The cloth that originally bear a fiber-feeling texture now looks like metallic/plastic.   

From this, it is clear that although during some periods of time, the proposed diffusion method achieves a lower relative entropy, it doesn’t necessarily imply that it has better visual appearance than FXAA-based methods - the metric itself is relatively ad-hoc, and its values themselves can’t directly translate to the quality of graphics. This is because it only captures the overall smoothness around the edges of the objects, not penalizing overly blurry edges (which, obviously, decreases the quality of the graphics). 

![**Figure:** Gradient entropy of different methods on the **arm** animation. Lower means less abrupt / jaggy edges.](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/entropy_arm.png)

**Figure:** Gradient entropy of different methods on the **arm** animation. Lower means less abrupt / jaggy edges.

![**Figure:** Gradient entropy of different methods on the **upper body** animation. Lower means less abrupt / jaggy edges.](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/entropy_upper_body.png)

**Figure:** Gradient entropy of different methods on the **upper body** animation. Lower means less abrupt / jaggy edges.

Therefore, instead of looking at the absolute values of the entropy, the closeness to the FXAA-based-method curve can be used, as it can be assumed to be the optimal graphics that hold the “perfect” balance between sharpness and smoothness. The L1-norm distance is measured for different methods compared to the FXAA-based method for their ability to closely follow the anti-aliasing profile generated by FXAA. As it can be shown from the plots below, the proposed diffusion (blue line) process very closely mirror the entropy behavior of the hardware-based FXAA method (red line), striking a balance between the unprocessed (jaggy) picture and the Gaussian processed (blurry) picture. At the same time, the original picture’s entropy curve is further away due to its jaggieness, whereas the orange curve (Gaussian) is even further away because it blurs out all the features.

![**Figure:** Relative entropy L-1 norm distance from the FXAA-based anti-aliasing profile curve near the **arm** area. Lower value is better.](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/relative_entropy_arm.png)

**Figure:** Relative entropy L-1 norm distance from the FXAA-based anti-aliasing profile curve near the **arm** area. Lower value is better.

![**Figure:** Relative entropy L-1 norm distance from the FXAA-based anti-aliasing profile curve in the **upper body** area. Lower value is better.](PDE-Based%20Anti-Aliasing%20in%20Computer%20Graphics%20e9d8473338d543a89fcdfdeed9776404/relative_entropy_upper_body.png)

**Figure:** Relative entropy L-1 norm distance from the FXAA-based anti-aliasing profile curve in the **upper body** area. Lower value is better.

The conclusion is therefore drawn that, this proposed diffusion process, out of all four methods, has the best anti-aliasing results that can arguably compare with the traditional super-sampling-based FXAA technique, suggesting that it is an effective way to perform anti-aliasing on graphics. 

# Part VI: Discussions

## High Level Comments

This formulation was inspired by the Perona-Malik diffusion which leverages weights to control the direction of diffusion. However, in contrast of what Perona-Malik does to an image - preserving edges and canceling out noises, this formulation does quite the opposite - smooths out edges while preserving textual features. It achieves this by assigning more weights (hence promoting diffusion) across edges, rather than along. 

This is a testament that these PDE formulations in image processing are flexible and general enough to achieve different goals in different scenarios, given the right tweaks.

## PDE Formulation

The over-simplified formulation in this project can be improved by better emulating the across-edge version of the Perona-Malik diffusion (Equation (2)). Examples of such optimizations include, and not limited to, using geometric heat diffusion instead of linear; using higher power on the gradient terms; using other terms (like Laplacian) to further penalize large gradients.

This will, of course, introduce additional computational cost because not only the combined central difference of the Laplacian, but also directional Laplacian and gradient values will need to be computed.

It is worth noting that the corresponding CFL condition might be trickier to derive.

## Selecting $\epsilon$

The threshold parameter $\epsilon$ shouldn’t be chosen arbitrarily. It can be deterministically computed given several hardware/nature-defined parameters:

1. Spatial resolution of human eyes
2. Spatial resolution and size of the display device
3. Color resolution of human eyes
4. Color resolution of display device

For example, Apple’s famous Retina Display specifically leveraged this collection of information to create displays that are exactly at the limit of human eye resolution. In this method, $\epsilon$ can be strategically chosen such that it smooths out jaggieness right towards the limit of human eye resolution, optimizing its computational cost while providing the most ideal appearance.

## Unavoidable Loss of Small Features

Because the diffusion does not leverage prior knowledge of the graphics itself, nor does it use super-sampling to enhance its understanding of the objects, it doesn’t require repeated computation of pixel values or extensive memory to store historical states. However, this advantage comes at a cost: because of its lack of such knowledge, it cannot distinguish jaggieness from “useful but small” features (like highlights in the character’s eyes). These very small features will be treated like jaggieness and smoothed out, causing loss of features.

One potential improvement is to create a hybrid method that combines traditional super-sampling with the proposed diffusion method, finding a middle-ground that balances computational cost and performance.

## Closing Words

In this short experiment, ways to use weighting to control the directions of diffusion processes are explored. The stability & CFL conditions are also derived and discussed. It has been proven that the diffusion equations are very flexible and can not only be used to preserve edges and cancel noises, but can also be used to perform exactly the opposite - preserving textures and removing jaggieness. This technique is purely software-based hence can run on any arbitrary machine. Also it is worth noting that the calculations are mostly local, hence after certain optimizations it should also be able to be transferred onto hardware level, achieving even better performance. Here, only preliminary research and results are presented - generalizations and optimizations will further exploit the potentials of this proposed technique.

# Appendix A: Full Code

```jsx
# importing the necessary libraries
import cv2
import numpy as np
from time import sleep, time
from numba import jit
from matplotlib import pyplot as plt

# low resolution version
# cap = cv2.VideoCapture('contrast_1280x720.mp4')

# high resolution version
cap = cv2.VideoCapture("contrast_1920x1080.mp4")

def inverse_perona_malik_diffusion(img, epsilon=1, b=50):
    # create a working copy.
    copy = img.copy()
    dx = 1000

    # compute regularized max gradient 
    gradient = cv2.Sobel(copy, cv2.CV_16S, 1, 0) + cv2.Sobel(copy, cv2.CV_16S, 0, 1)
    max_grad = np.max(gradient / 1e3)

    # enter gradient descent
    while max_grad > epsilon:

        # compute directional derivatives and second derivatives 
        Ix = cv2.Sobel(copy, cv2.CV_32F, 1, 0) / dx
        Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0) / dx
        Iy = cv2.Sobel(copy, cv2.CV_32F, 0, 1) / dx
        Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1) / dx

        # compute update 
        It = b * dx * ((Ix**2 + Iy**2)**2) * (Ixx + Iyy)
        diffused = (copy + It)
        copy = diffused.astype("uint8")

        # recompute regularized max gradient
        gradient = cv2.Sobel(copy, cv2.CV_16S, 1, 0) + cv2.Sobel(copy, cv2.CV_16S, 0, 1)
        max_grad = np.max(gradient / 1e3)
        # print("gradient: ", max_grad)
    return copy

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
left = 700
top = 200
offset = 1920
grid_size = 500

output_size = (2 * grid_size, 2 * grid_size)
out = cv2.VideoWriter(
    "output.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 30, output_size
)

# define and record entropies of images
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

    # put frames all together side-by-side
    frame = np.concatenate(
        (
            np.concatenate([diffused, original], axis=1), 
            np.concatenate([gaussian, hardware_antialiased], axis=1)
        ),
        axis=0)
    frame = cv2.resize(frame, output_size)

    # compute and record entropies
    diffused_entropy = gradient_square_entropy(diffused)
    original_entropy = gradient_square_entropy(original)
    gaussian_entropy = gradient_square_entropy(gaussian)
    hardware_entropy = gradient_square_entropy(hardware_antialiased)
    entropies['diffused'].append(diffused_entropy)
    entropies['original'].append(original_entropy)
    entropies['gaussian'].append(gaussian_entropy)
    entropies['hardware'].append(hardware_entropy)

    # put relative entropies onto frame
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

# plot the entropy evolution
t = [i for i in range(len(entropies['diffused']))]
for (key, val) in entropies.items():
    plt.plot(t, val, label=key)
plt.legend()
plt.show()
```