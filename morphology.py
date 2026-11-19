import numpy as np

# Morphological operations from TASK 5 - used for background correction (top-hat)
# and for cleaning the binary mask (opening / closing)

# WŁASNE: define the structuring element (SE) for morphological operations 
#           two options: square and disk
def make_structuring_element(shape: str, size: int):
    if shape == "Square":
        return np.ones((size, size), dtype=bool)
    elif shape == "Disk":
        radius = size // 2
        se = np.zeros((size, size), dtype=bool)
        for i in range(size):
            for j in range(size):
                if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                    se[i, j] = 1
        return se
    else:
        raise ValueError("Unsupported structuring element shape. Use 'Square' or 'Disk'.")

# WŁASNE: define erosion based on previous task adaptive thresholding logic (same windowing logic)
# def erosion(image: np.ndarray, se: np.ndarray):
#     window_radius = se.shape[0] // 2
#     padded_image = np.pad(image, window_radius, mode='edge')
#     eroded = np.zeros(image.shape)
#     height, width = image.shape
#
#     for x in range(height):
#         for y in range(width):
#             window = padded_image[x:x + se.shape[0], y:y + se.shape[1]]
#             eroded[x, y] = np.min(window[se])
#
#     return eroded

# GENAI: Claude – optimized erosion: the loop over pixels is replaced with a loop over SE elements
def optimized_erosion(image: np.ndarray, se: np.ndarray):
    window_radius = se.shape[0] // 2
    padded_image = np.pad(image, window_radius, mode='edge').astype(np.float64)
    height, width = image.shape
    eroded = np.full(image.shape, np.inf)

    for n, m in zip(*np.nonzero(se)):    # positions of True elements of the SE
        piece_of_image = padded_image[n:n + height, m:m + width]
        eroded = np.minimum(eroded, piece_of_image)

    return eroded

# WŁASNE: define dilation based on previous task adaptive thresholding logic (same windowing logic)
# def dilation(image: np.ndarray, se: np.ndarray):
#     window_radius = se.shape[0] // 2
#     padded_image = np.pad(image, window_radius, mode='edge')
#     dilated = np.zeros(image.shape)
#     height, width = image.shape
#
#     for x in range(height):
#         for y in range(width):
#             window = padded_image[x:x + se.shape[0], y:y + se.shape[1]]
#             dilated[x, y] = np.max(window[se])
#
#     return dilated

# GENAI: Claude – optimized dilation: same idea as optimized_erosion, with np.maximum
def optimized_dilation(image: np.ndarray, se: np.ndarray):
    window_radius = se.shape[0] // 2
    padded_image = np.pad(image, window_radius, mode='edge').astype(np.float64)
    height, width = image.shape
    dilated = np.full(image.shape, -np.inf)

    for n, m in zip(*np.nonzero(se)):
        piece_of_image = padded_image[n:n + height, m:m + width]
        dilated = np.maximum(dilated, piece_of_image)

    return dilated

# WŁASNE: opening defined = erosion followed by dilation
def opening(image: np.ndarray, se: np.ndarray):
    eroded = optimized_erosion(image, se)
    opened = optimized_dilation(eroded, se)

    return opened

# WŁASNE: closing defined = dilation followed by erosion
def closing(image: np.ndarray, se: np.ndarray):
    dilated = optimized_dilation(image, se)
    closed = optimized_erosion(dilated, se)

    return closed
