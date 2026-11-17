import numpy as np

# Steps 3-4: cutting objects out, orientation, rotation, shape descriptors


# TODO (WŁASNE): step 3.1
def crop_object(image: np.ndarray, labels: np.ndarray, label: int, margin: int):
    """
    image:  H x W x 3, uint8 (RGB)
    labels: H x W, int (result of connected_component_labeling)
    label:  number of the object to cut out
    margin: extra pixels around the bounding box
    returns: (image_crop, mask_crop)
             image_crop - h x w x 3 fragment of the image
             mask_crop  - h x w, bool or 0/1 - ONLY this object (not its neighbours inside the box)
    """
    raise NotImplementedError("crop_object (geometry.py)")


# TODO (WŁASNE): step 3.2
def object_orientation(mask: np.ndarray):
    """
    mask: h x w, bool or 0/1
    returns: theta - angle of the main axis w.r.t. the horizontal axis, in DEGREES
    """
    raise NotImplementedError("object_orientation (geometry.py)")


# TODO (WŁASNE): step 3.3
def rotate_object(image_crop: np.ndarray, mask_crop: np.ndarray, theta: float):
    """
    image_crop: h x w x 3 (bilinear interpolation; tip: the weights can be broadcast over the 3 channels)
    mask_crop:  h x w (nearest neighbour)
    theta:      angle from object_orientation, in degrees
    returns: (image_rotated, mask_rotated) - main axis horizontal, output big enough not to cut the object
    """
    raise NotImplementedError("rotate_object (geometry.py)")


# TODO (WŁASNE): step 4
def shape_descriptors(mask_original: np.ndarray, mask_rotated: np.ndarray):
    """
    mask_original: h x w object mask before rotation
    mask_rotated:  object mask after rotation (decide which one you use for which descriptor - README)
    returns: dict with keys
             "area", "perimeter", "width", "height", "aspect_ratio", "rectangularity", "circularity"
    """
    raise NotImplementedError("shape_descriptors (geometry.py)")
