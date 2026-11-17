from collections import deque

import numpy as np

# lepiej zaczac od tych bez polaczen - te z polaczeniami bardzo lekkimi moga zostac wykryte przez erozje i naprawione przez dylacje.
# gorzej z wiekszymi sklejeniami gdzie raczej nie ma przerw miedzy ziarenkami 
# ofc mozna uzyc czegos innego niz ziarenka 
def connected_component_labeling(mask: np.ndarray, connectivity: int = 8):
    binary_mask = (mask > 0).astype(np.uint8) 

    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 0

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
    else:
        raise ValueError("Unsupported connectivity. Use 4 or 8.")

    queue = deque()

    for y in range(height):
        for x in range(width):
            if binary_mask[y,x] == 1 and labels[y, x] == 0:
                current_label += 1
                labels[y, x] = current_label

                queue.append((y, x))

                while queue:
                    current_y, current_x = queue.popleft()

                    for delta_y, delta_x in neighbors:
                        neighbour_y = current_y + delta_y
                        neighbour_x = current_x + delta_x

                        if (0 <= neighbour_y < height) and (0 <= neighbour_x < width):
                            if binary_mask[neighbour_y, neighbour_x] == 1 and labels[neighbour_y, neighbour_x] == 0:
                                labels[neighbour_y, neighbour_x] = current_label
                                queue.append((neighbour_y, neighbour_x))

    return labels, current_label



# TODO (WŁASNE): step 1
def labels_to_color(labels: np.ndarray, n: int):
    """
    labels: H x W, int, 0 = background, 1..n = objects
    returns: H x W x 3, uint8 - a different color for every object, background black
    """
    raise NotImplementedError("labels_to_color (objects_analysis.py)")

# TODO (WŁASNE): step 2
def compute_areas(labels: np.ndarray, n: int):
    """
    returns: 1D array of length n + 1, areas[i] = number of pixels of object i
             (areas[0] = background, ignored)
    """
    raise NotImplementedError("compute_areas (objects_analysis.py)")

# TODO (WŁASNE): step 2
def classify_objects(areas: np.ndarray, eps: float, k: float):
    """
    areas: result of compute_areas (index = label, areas[0] ignored)
    eps:   relative tolerance for single objects
    k:     threshold coefficient for connected objects (k > 1 + eps)
    returns: (categories, mean_area)
             categories - list of length n + 1, categories[i] is "single", "connected" or "other"
                          (categories[0] ignored)
             mean_area  - the mean area used for the classification
    """
    raise NotImplementedError("classify_objects (objects_analysis.py)")
