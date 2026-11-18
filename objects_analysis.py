from collections import deque
import numpy as np

# WŁASNE: labeling the detected objects with BFS
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

# WŁASNE: coloring object labels as groups
def labels_to_color(labels: np.ndarray, n: int):
    color_generator = np.random.default_rng(0)
    size = (n + 1, 3)
    colors = color_generator.integers(50, 256, size=size)
    colors = np.array(colors, dtype=np.uint8)
    colors[0] = 0

    return colors[labels]

# WŁASNE: position number = label number, value = how many pixels does this label have
# logic based on previous histogram calculations
def compute_areas(labels: np.ndarray, n: int):
    label_bins = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)

    return label_bins
    
# WŁASNE: classifying objects based on their calculated area
def classify_objects(areas: np.ndarray, eps: float, k: float):
    N = len(areas) - 1 # areas[0] is the background
    categories = ["background"]

    if N == 0:
        return categories, 0
    
    average_area = areas[1:].sum() * (1/N)

    for i in range(1, N + 1):
        if average_area * (1 - eps) <= areas[i] <= average_area * (1 + eps):
            categories.append("single")
        elif areas[i] > k * average_area:
            categories.append("connected")
        else:
            categories.append("other")

    return categories, average_area

