import numpy as np
from PIL import Image
from image_segmentation import apply_threshold, optimized_otsu_threshold
from background_removal import blur_background, white_tophat_background, black_tophat_background, correction_to_uint8
from morphology import make_structuring_element, opening, closing


# WŁASNE: prepare mask for object extraction
def compute_mask(image_rgb: np.ndarray, channel: str, dark_objects: bool, background: str, background_param, se_size: int):
    # 1. channel -> 2D intensity image
    if channel == "Blue":
        intensity = image_rgb[..., 2]
    else:
        pil_image = Image.fromarray(image_rgb)
        pil_image_greyscale = pil_image.convert("L")
        intensity = np.array(pil_image_greyscale)

    # 2. background correction -> uint8 image for thresholding
    if background == "None":
        to_threshold = intensity
    elif background == "Gaussian blur":
        radius = int(np.ceil(3 * background_param))
        size = 2 * radius + 1
        _, R = blur_background(intensity, background_param, size)
        to_threshold = correction_to_uint8(R, 128)
    elif background == "Top-hat":
        if dark_objects:
            _, R = black_tophat_background(intensity, "Disk", background_param)
            to_threshold = correction_to_uint8(R, 0)
        else:
            _, R = white_tophat_background(intensity, "Disk", background_param)
            to_threshold = correction_to_uint8(R, 0)
    else:
        raise ValueError("Unsupported background correction. Use 'None', 'Gaussian blur' or 'Top-hat'.")
    
    # 3. Otsu threshold
    t, _ = optimized_otsu_threshold(to_threshold)
    mask = apply_threshold(to_threshold, t)
    
    # 4. threshold direction -> binary mask
    if dark_objects and (background != "Top-hat"):
        mask = 255 - mask

    # 5. cleaning (opening / closing) if se_size > 0
    if se_size > 0:
        se = make_structuring_element("Disk", se_size)
        opened = opening(mask, se)
        closed = closing(opened, se)

        mask = closed.astype(np.uint8)

    # 6. return mask (uint8, 0/255) and t
    return mask, t