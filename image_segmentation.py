import numpy as np

# Code from TASK 4 for image segmentation

# WŁASNE: calculating image histogram
# def calculate_image_histogram(image: np.ndarray):
#     height, width = image.shape
#     N = height * width
#
#     # 1. Calculate image histogram
#     histogram_bins = np.zeros(256)
#     for i in range (0, height):
#         for j in range (0, width):
#             intensity = image[i,j]
#             histogram_bins[intensity] += 1
#
#     # 2. Normalize to the probability distribution
#     probability = np.zeros(256)
#     for k in range(256):
#         probability[k] = histogram_bins[k] / N
#
#     return probability, histogram_bins

# GENAI: Claude - optimized version of own calculate_image_histogram (task 4)
def optimized_calculate_image_histogram(image: np.ndarray):
    N = image.size

    # 1. Calculate image histogram: np.bincount counts how many times each value 0..255 occurs
    histogram_bins = np.bincount(image.ravel(), minlength=256).astype(np.float64)

    # 2. Normalize to the probability distribution (whole array at once)
    probability = histogram_bins / N

    return probability, histogram_bins

# WŁASNE: applying the threshold to the image (creating the binary mask)
def apply_threshold(image: np.ndarray, t: int):
    # M(x,y) = 1 if I(x,y) >= t else 0
    mask = image >= t
    return mask.astype(np.uint8) * 255

# WŁASNE: otsu thresholding algorithm based on the instructions provided in the task
# def otsu_threshold(image: np.ndarray):
#     probability, _ = calculate_image_histogram(image)
#
#     sigma_b_squared = np.zeros(256)
#
#     # for each possible threshold t calculate the between-class variance
#     for t in range(256):
#         omega_0 = 0.0
#         mi_0 = 0.0
#         mi_1 = 0.0
#
#         for i in range(256):
#             if i <= t:
#                 omega_0 += probability[i]
#                 mi_0 += i * probability[i]
#             else:
#                 mi_1 += i * probability[i]
#
#         omega_1 = 1 - omega_0
#
#         # skip if either class is empty (to avoid division by zero)
#         if omega_0 > 0 and omega_1 > 0:
#             mi_0 = mi_0 / omega_0
#             mi_1 = mi_1 / omega_1
#
#             sigma_b_squared[t] = omega_0 * omega_1 * (mi_0 - mi_1) ** 2
#
#     best_threshold = int(np.argmax(sigma_b_squared))
#
#     return best_threshold, sigma_b_squared

# GENAI: Claude - optimized version of otsu_threshold (task 4)
def optimized_otsu_threshold(image: np.ndarray):
    probability, histogram_bins = optimized_calculate_image_histogram(image)

    sigma_b_squared = np.zeros(256)

    i = np.arange(256)
    omega_0 = np.cumsum(probability)              # omega_0[t] = sum_{i<=t} p(i)
    omega_1 = 1 - omega_0
    mi_0_sum = np.cumsum(i * probability)         # sum_{i<=t} i*p(i)
    mi_1_sum = mi_0_sum[-1] - mi_0_sum            # sum_{i>t} i*p(i) = total - sum_{i<=t}

    # skip t where either class is empty (to avoid division by zero)
    pixels_0 = np.cumsum(histogram_bins)
    valid = (pixels_0 > 0) & (pixels_0 < image.size)

    mi_0 = mi_0_sum[valid] / omega_0[valid]
    mi_1 = mi_1_sum[valid] / omega_1[valid]

    sigma_b_squared[valid] = omega_0[valid] * omega_1[valid] * (mi_0 - mi_1) ** 2

    best_threshold = int(np.argmax(sigma_b_squared))

    return best_threshold, sigma_b_squared