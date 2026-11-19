import numpy as np

# WŁASNE: bounding boxes for object cropping
def crop_object(image: np.ndarray, labels: np.ndarray, label: int, margin: int):
    # 1. pixels of this object -> their rows (y) and columns (x)
    object_to_crop = labels == label
    y, x = np.nonzero(object_to_crop)

    # 2. bounding box: x_min, x_max, y_min, y_max + add margin and clip to image borders
    height, width = labels.shape
    x_min = max(x.min() - margin, 0)
    y_min = max(y.min() - margin, 0)
    x_max = min(x.max() + margin, width - 1)
    y_max = min(y.max() + margin, height - 1) 

    # 3. cut out the image fragment and the mask of ONLY this object
    image_object_fragment = image[y_min : y_max + 1, x_min : x_max + 1]
    mask_object_fragment = object_to_crop[y_min : y_max + 1, x_min : x_max + 1]

    return image_object_fragment, mask_object_fragment

# WŁASNE: calculating object's orientation
def object_orientation(mask: np.ndarray):
    # 1. coordinates of object pixels (x = column, y = row) - we take those pixels of M that are = 1
    y, x = np.nonzero(mask)

    # 2. centroid x̄, ȳ
    x_center_of_gravity = x.mean()
    y_center_of_gravity = y.mean()

    # 3. central moments μ20, μ02, μ11
    rows = y - y_center_of_gravity
    columns =  x - x_center_of_gravity
    mi_20 = (columns ** 2).sum()
    mi_02 = (rows ** 2).sum()
    mi_11 = (columns * rows).sum()

    # 4. θ = ½ · atan2(2μ11, μ20 − μ02)
    theta = 1/2 * np.arctan2(2 * mi_11, mi_20 - mi_02)

    # 5. return θ in degrees
    return np.degrees(theta)

# WŁASNE: bilinear interpolation formula copied from the previous tasks
def bilinear_formula(u, v, v_00, v_10, v_01, v_11):
    return (1 - u)*(1 - v)*v_00 + u*(1 - v)*v_10 + (1 - u)*v*v_01 + u*v*v_11

# GENAI: Claude - helped with the correct calculations and explaining the logic behind them
# ESPECIALLY: adjusting the bilinear interpolation logic of the previous task's version to fit this new task
def rotate_object(image_crop: np.ndarray, mask_crop: np.ndarray, theta: float):
    # 1. angle in radians, cos and sin
    theta = np.radians(theta)

    # 2. output size (big enough not to cut the object)
    h, w = mask_crop.shape
    W = int(np.ceil(w * abs(np.cos(theta)) + h * abs(np.sin(theta))))
    H = int(np.ceil(w * abs(np.sin(theta)) + h * abs(np.cos(theta))))

    # 3. coordinates of every output pixel, relative to the output centre
    y, x = np.indices((H, W))
    dy_out = y - (H - 1) / 2
    dx_out = x - (W - 1) / 2

    # 4. inverse mapping -> source coordinates
    cx = (w - 1) / 2
    cy = (h - 1) / 2
    x_src = cx + np.cos(theta) * dx_out - np.sin(theta) * dy_out
    y_src = cy + np.sin(theta) * dx_out + np.cos(theta) * dy_out

    # 5. mask: nearest neighbour
    x_i = (x_src.round()).astype(int)
    y_i = (y_src.round()).astype(int)

    inside = (0 <= x_i) & (x_i <= w - 1) & (0 <= y_i) & (y_i <= h - 1)
    mask_rotated = np.zeros((H, W), dtype=bool)
    mask_rotated[inside] = mask_crop[y_i[inside], x_i[inside]]

    # 6. image: bilinear interpolation
    # GENAI: Claude – step 6 (bilinear sampling of the RGB crop at the non-integer source coordinates)
    # 6a. the 4 neighbouring pixels: (x0, y0) top-left, (x1, y1) bottom-right
    x0 = np.floor(x_src).astype(int)
    y0 = np.floor(y_src).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # 6b. fractional parts = position of the point inside the 2 x 2 square (0..1)
    u = x_src - x0
    v = y_src - y0

    # 6c. all 4 neighbours must lie inside the source image
    inside_bilinear = (0 <= x0) & (x1 <= w - 1) & (0 <= y0) & (y1 <= h - 1)
    x0, x1, y0, y1 = x0[inside_bilinear], x1[inside_bilinear], y0[inside_bilinear], y1[inside_bilinear]

    # 6d + 6e. formula from task 5; [:, None] -> one weight for all 3 channels of a pixel
    source = image_crop.astype(np.float64)
    values = bilinear_formula(u[inside_bilinear][:, None], v[inside_bilinear][:, None],
                              source[y0, x0], source[y0, x1], source[y1, x0], source[y1, x1])

    # 6f. white background (as in the photos), interpolated values where the source exists
    image_rotated = np.full((H, W, 3), 255.0)
    image_rotated[inside_bilinear] = values
    image_rotated = np.clip(np.round(image_rotated), 0, 255).astype(np.uint8)

    # 7. return
    return image_rotated, mask_rotated

# GENAI: Claude – perimeter by Moore neighbour contour tracing:
#   walk around the object along its boundary ("right hand on the wall" = background) and sum the
#   lengths of the steps between consecutive contour pixels: 1 for a straight step, sqrt(2) for a diagonal one.
#   Stop (Jacob's criterion): back at the start pixel AND about to repeat the very first step - plain
#   "back at the start" would stop halfway on thin shapes, where the contour passes the start pixel twice.
def contour_perimeter(mask: np.ndarray):
    # 1. frame of background around the mask -> neighbours never fall outside the array
    padded = np.pad(mask.astype(bool), 1)

    # 2. 8 directions (dy, dx), clockwise starting from east; even index = straight, odd = diagonal
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    #              E       SE      S       SW       W        NW        N        NE

    # 3. start pixel: top-most, then left-most (np.nonzero goes row by row) -> above and left of it is background
    ys, xs = np.nonzero(padded)
    if ys.size == 0:
        return 0.0
    start = (ys[0], xs[0])

    # 4. walk around the object
    current = start
    last_direction = 0          # direction of the last step
    first_direction = None      # direction of the very first step (for the stop condition)
    perimeter = 0.0

    while True:
        # search clockwise, starting 90° to the left of the last step (where the background should be)
        search_start = (last_direction + 6) % 8
        next_direction = None
        for j in range(8):
            d = (search_start + j) % 8
            dy, dx = directions[d]
            if padded[current[0] + dy, current[1] + dx]:
                next_direction = d
                break

        if next_direction is None:          # isolated single pixel
            return 0.0

        # 5. stop: at the start again and about to repeat the first step
        if current == start and next_direction == first_direction:
            break
        if first_direction is None:
            first_direction = next_direction

        dy, dx = directions[next_direction]
        current = (current[0] + dy, current[1] + dx)
        last_direction = next_direction
        perimeter += 1.0 if next_direction % 2 == 0 else np.sqrt(2)

    # 6. sum of step lengths
    return perimeter

# WŁASNE: shape descriptors defined (except help with perimeter function as above)
def shape_descriptors(mask_original: np.ndarray, mask_rotated: np.ndarray):
    area = (mask_original == True).sum()

    y, x = np.nonzero(mask_rotated)
    h, w = mask_rotated.shape
    x_min = max(x.min(), 0)
    y_min = max(y.min(), 0)
    x_max = min(x.max(), w - 1)
    y_max = min(y.max(), h - 1)
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    aspect_ratio = width / height

    rectangularity = area / (width * height)

    perimeter = contour_perimeter(mask_original)

    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan

    return {"area" : area,
            "perimeter": perimeter,
            "width" : width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "rectangularity" : rectangularity,
            "circularity": circularity}






