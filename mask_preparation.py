import numpy as np

# Mask computed inside the program: optional background correction (task 5),
# thresholding (task 4) and morphological cleaning (task 5)


# TODO (WŁASNE): step 0
def compute_mask(image_rgb: np.ndarray, channel: str, dark_objects: bool, background: str, background_param, se_size: int):
    """
    image_rgb:        H x W x 3, uint8 (RGB)
    channel:          "Gray" or "Blue" - which intensity image is used
    dark_objects:     True  - objects darker than the background (rice, seeds),
                      False - objects brighter than the background (vignetting_objects.png)
    background:       background correction before thresholding:
                      "None", "Gaussian blur" or "Top-hat"
    background_param: None for "None", sigma for "Gaussian blur", SE size (odd) for "Top-hat"
    se_size:          size of the structuring element used to clean the mask, 0 = no morphology
    returns:          (mask, t)
                      mask - H x W, uint8, values 0 (background) / 255 (object)
                      t    - threshold that was used (shown in the GUI)
    """
    raise NotImplementedError("compute_mask (mask_preparation.py)")
