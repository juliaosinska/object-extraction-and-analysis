import numpy as np
from morphology import make_structuring_element, opening, closing

# Code from TASK 5 for background correction and removal

# ------------------------
# Gaussian blur background estimation
# ------------------------

# WŁASNE: from previous tasks - image filtering
# GENAI: Claude – adaptation of my task-3 Gaussian code to 2D grayscale arrays (no uint8 rounding between passes)
# WŁASNE: adapted to task 5: added size parameter to get_separable_gaussian_kernel()
#           and gaussian_filter() to allow for different kernel sizes

def get_separable_gaussian_kernel(sigma=1.0, size=3):
    radius = size // 2

    x = np.arange(-radius, radius + 1)
    # GENAI: Gemini used to help with using correct Gaussian kernel formula
    kernel = np.exp(-(x**2) / (2 * (sigma**2)))

    kernel /= np.sum(kernel)

    return kernel

def optimized_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # image: 2D float array, kernel: 2D array; returns 2D float array (same size as image)
    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    output = np.zeros(image.shape, dtype=np.float64)

    # GENAI: Gemini used to help with optimizing the convolution operation by iterating over the kernel and applying it to the entire image at once
    for n in range(kernel_height):
        for m in range(kernel_width):
            weight = kernel[n, m]

            if weight == 0:
                continue

            piece_of_image = padded_image[n:n + image_height, m:m + image_width]
            output += piece_of_image * weight

    return output

def gaussian_filter(image: np.ndarray, sigma: float = 1.0, size: int = 3) -> np.ndarray:
    kernel = get_separable_gaussian_kernel(sigma=sigma, size=size)
    kernel_size = len(kernel)

    kernel_h = kernel.reshape((1, kernel_size))
    kernel_v = kernel.reshape((kernel_size, 1))

    h_convolved = optimized_convolution(image.astype(np.float64), kernel_h)
    final = optimized_convolution(h_convolved, kernel_v)

    return final

# WŁASNE: using Gaussian blur filter to estimate B and R
def blur_background(image: np.ndarray, sigma: float, size: int):
    B = gaussian_filter(image.astype(np.float64), sigma=sigma, size=size)
    # here B(x,y) = Gσ(I)(x,y)
    R = image - B

    return B, R

# ------------------------
# White/Black Top-hat background estimation
# ------------------------

# WŁASNE: define tophat background estimation using morphological opening
# ~ 4x faster thanks to optimized erosion and dilation
def white_tophat_background(image: np.ndarray, se_shape: str, se_size: int):
    se = make_structuring_element(se_shape, se_size)
    B = opening(image, se)
    R = image - B

    return B, R

# WŁASNE: define tophat background estimation using morphological closing
def black_tophat_background(image: np.ndarray, se_shape: str, se_size: int):
    se = make_structuring_element(se_shape, se_size)
    B = closing(image, se)
    R = B - image

    return B, R

# ------------------------
# Preparing R for thresholding
# ------------------------

# GENAI: Claude – shifting and clipping R to uint8 0..255 before Otsu
# R is float and can be negative; Otsu (np.bincount) needs 0..255 -> shift by offset (blur: 128, top-hat: 0) and clip
def correction_to_uint8(R: np.ndarray, offset: float = 0.0) -> np.ndarray:
    return np.clip(np.round(R + offset), 0, 255).astype(np.uint8)
