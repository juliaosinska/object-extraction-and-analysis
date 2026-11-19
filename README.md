# Object Extraction and Analysis

A small Python desktop app (Tkinter) that finds objects in an image, labels them, classifies them by size, aligns the single objects to a common orientation and describes their shape with geometric descriptors. The binary mask is computed inside the program (thresholding from task 4, optional background correction and morphology from task 5); a ready mask can also be loaded from a file.

Repository: [juliaosinska/object-extraction-and-analysis](https://github.com/juliaosinska/object-extraction-and-analysis)

## Features

- Load an image (RGB) and, optionally, a binary mask (8-bit: 0 = background, any other value = object, same size as the image)
- **Mask computation**: gray or blue channel, objects darker or brighter than the background, optional background correction (Gaussian blur or top-hat), Otsu threshold, mask cleaning with opening and closing
- **Connected component labeling** (own BFS implementation) with 4- or 8-connectivity, label image with a different color for every object, side-by-side comparison of both connectivities
- **Area-based classification** into single, connected and other objects, with adjustable tolerance ε and coefficient k (the GUI rejects k ≤ 1 + ε)
- **Classification overlay**: input image with single objects in green, connected in red, other in yellow, a legend with the number of objects in each category and the label number of every object
- **Cutting out and alignment**: bounding box with margin, orientation from central moments, rotation to a horizontal main axis (bilinear interpolation for the image, nearest neighbour for the mask), grid of all aligned single objects at a common scale
- **Shape descriptors**: area, perimeter (contour tracing), width, height, aspect ratio, orientation angle, rectangularity, circularity
- **Table of object parameters** (sortable by any column) and **size statistics** of single objects (mean, standard deviation, min, max) with a histogram of areas and the classification limits
- **Effect of ε and k**: number of objects in each category for a range of ε and k
- **Shape comparison**: rectangularity and aspect ratio plotted against circularity for all objects

## Methods

| Step | Formula / rule | Implementation |
|---|---|---|
| Mask | `M = 1 if I < t else 0` for dark objects, `I ≥ t` for bright ones; `t` from Otsu | `compute_mask` |
| Background correction (optional) | blur: `R = I − Gσ(I)`; white top-hat: `R = I − (I ∘ SE)`; black top-hat: `R = (I • SE) − I` | `blur_background`, `white_tophat_background`, `black_tophat_background` |
| Mask cleaning (optional) | opening `(M ⊖ SE) ⊕ SE`, then closing `(M ⊕ SE) ⊖ SE`, disk SE | `opening`, `closing` |
| Labeling | BFS flood fill; neighbours: 4 (N, S, W, E) or 8 (+ diagonals) | `connected_component_labeling` |
| Area | `Aᵢ` = number of pixels with label `i` | `compute_areas` (`np.bincount`) |
| Mean area | `Ā = (1/N) · Σ Aᵢ` (background excluded) | `classify_objects` |
| Single | `Ā(1 − ε) ≤ Aᵢ ≤ Ā(1 + ε)` | `classify_objects` |
| Connected | `Aᵢ > k · Ā` | `classify_objects` |
| Other | neither of the above | `classify_objects` |
| Bounding box | `x_min, x_max, y_min, y_max` of the object pixels, + margin, clipped to the image | `crop_object` |
| Centroid | `x̄ = Σx / A`, `ȳ = Σy / A` | `object_orientation` |
| Central moments | `μ20 = Σ(x − x̄)²`, `μ02 = Σ(y − ȳ)²`, `μ11 = Σ(x − x̄)(y − ȳ)` | `object_orientation` |
| Orientation | `θ = ½ · atan2(2μ11, μ20 − μ02)` | `object_orientation` |
| Rotation | inverse mapping: `x_src = cx + cos θ · dx − sin θ · dy`, `y_src = cy + sin θ · dx + cos θ · dy` | `rotate_object` |
| Bilinear interpolation | `(1−u)(1−v)·v00 + u(1−v)·v10 + (1−u)v·v01 + uv·v11` (same formula as in task 5) | `bilinear_formula` |
| Perimeter | sum of the steps along the traced contour: 1 (straight), √2 (diagonal) | `contour_perimeter` |
| Rectangularity | `A / (W · H)` | `shape_descriptors` |
| Circularity | `4πA / P²` | `shape_descriptors` |
| Aspect ratio | `W / H` | `shape_descriptors` |

Details and decisions:

- **M(x, y) in the moment formulas**: M is 0 or 1, so every sum `Σ f(x, y) · M(x, y)` is a sum over the object pixels only. They are taken with `np.nonzero(mask)`.
- **Orientation angle**: in degrees, in the range −90°…90°, measured from the horizontal axis. The image y axis points down, so a positive angle means the object goes "right and down" (clockwise on the screen).
- **Rotation**: the output size is `W' = w|cos θ| + h|sin θ|`, `H' = w|sin θ| + h|cos θ|`, so the object is never cut. The background of the rotated image is white (as in the photos).
- **Which mask for which descriptor**: area and perimeter come from the **original** mask (the nearest-neighbour rotation changes the number of pixels slightly and makes the edges jagged). Width, height, aspect ratio and rectangularity come from the bounding box of the **rotated** mask, so they do not depend on how the object lay in the image (after the rotation W is the length along the main axis and H the thickness).
- **Perimeter**: Moore neighbour contour tracing on the mask padded with background. It stops when it is back at the start pixel and about to repeat the first step (Jacob's criterion); stopping simply at the start pixel would end halfway on thin shapes, where the contour passes the start pixel twice. Single pixels have perimeter 0 and an undefined circularity.
- **Channel**: the gray image is `convert("L")` (ITU-R BT.601 weights 0.299 / 0.587 / 0.114, as in the previous tasks); the blue channel is `I[..., 2]`.
- **Background correction before Otsu**: R is a float image and can be negative, but Otsu (`np.bincount`) needs integers 0…255, so R is shifted (blur: +128, top-hat: 0) and clipped. After a black top-hat dark objects become bright in R, so the threshold direction is `R ≥ t` for both top-hats and `I < t` only for dark objects without top-hat.

## Project structure

- `gui.py`: Tkinter GUI, all views, table and plots (matplotlib)
- `mask_preparation.py`: `compute_mask` (channel, background correction, Otsu, threshold direction, cleaning)
- `objects_analysis.py`: connected component labeling, label colors, areas, classification
- `geometry.py`: cutting out, orientation, rotation, bilinear interpolation, contour perimeter, shape descriptors
- `image_segmentation.py`: histogram, `apply_threshold`, Otsu (from task 4)
- `background_removal.py`: Gaussian blur, white and black top-hat, `correction_to_uint8` (from task 5, black top-hat added)
- `morphology.py`: structuring element, erosion, dilation, opening, closing (from task 5, closing added)
- `test_images/`: test images

## Requirements and running

Python 3 with NumPy, Pillow and matplotlib (Tkinter is included with the standard Windows installer):

```powershell
py -m pip install numpy Pillow matplotlib
py gui.py
```

Load an image, set the parameters and click *Process Image*; the view is chosen from the *View* list. Defaults: blue channel, dark objects, no background correction, cleaning SE 3, 8-connectivity, ε = 0.2, k = 1.5, margin 5 px.

## Notes

- AI usage:
    - GUI implementation (`gui.py` file),
    - `correction_to_uint8` (shifting and clipping R before Otsu) - effect: Otsu (`np.bincount`) also works on the background-corrected image R, which is float and has negative values,
    - help with the inverse mapping in `rotate_object` and the bilinear sampling of the RGB image (step 6), adapting the bilinear formula from task 5 to arrays - effect: rotated objects without holes, never cut off, with smooth edges (checked against `Image.rotate` from Pillow: mean difference below 0.5 / 255),
    - `contour_perimeter` (Moore neighbour contour tracing) - effect: perimeter as the sum of steps along the ordered contour, as required; the stop condition gives the full contour also for thin shapes, where a simple "back at the start" check stops halfway,
    - optimized versions of histogram, Otsu, erosion and dilation reused from tasks 4 and 5 (own loop versions kept in the code, commented out above them).

- Observations:
    - **Channel**: the grains are yellowish on a white background, so they stand out most in the blue channel (about twice the contrast of the gray image, where blue has the smallest weight). The blue channel gives a much cleaner mask (seeds: 17 objects instead of 65).

      <img width="1397" height="1267" alt="image" src="https://github.com/user-attachments/assets/dae5dc6e-b0dc-4273-8e66-7e587684960d" />
      <img width="1395" height="1265" alt="image" src="https://github.com/user-attachments/assets/3739e3ab-3c74-4038-8141-c6cf4e56f475" />

    - **4- vs 8-connectivity**: 4-connectivity splits pixels that touch only diagonally, so it gives more objects, almost all of them tiny fragments (rice without cleaning: 43 vs 27). The large objects are nearly the same, and after mask cleaning the difference becomes small.

      <img width="1399" height="1270" alt="image" src="https://github.com/user-attachments/assets/08d17bf6-4395-480c-a903-e796cf58dc57" />

    - **Mask cleaning**: opening and closing with a small SE (3) remove the fragments. A larger SE also merges grains that lie close together, because the closing fills the gap between them.

    - **Background correction**: necessary for uneven illumination (vignetting image: 803 objects without correction, 59 with it), but harmful for a uniform background: it pulls the grains into the background estimate and leaves mostly their shadows and edges. Top-hat also loses objects larger than the SE. Therefore it is off by default.

      <img width="1397" height="1265" alt="image" src="https://github.com/user-attachments/assets/a144134a-6b92-46e9-a522-e1ee9f6fe946" />
      <img width="1397" height="1266" alt="image" src="https://github.com/user-attachments/assets/d2be84b9-1fe4-484a-a6c8-bf85c0d83635" />

    - **Effect of ε and k**: a larger ε moves more objects into "single", a larger k moves smaller clusters from "connected" to "other". On seeds (similar grains) the result is stable from ε = 0.2. On rice a few large clusters raise the mean to about the area of two grains, so with ε = 0.2 most single grains end up in "other" and a pair of grains is "single". The area-based classification is approximate, because the mean is not robust to a few large objects.

      <img width="1398" height="1387" alt="image" src="https://github.com/user-attachments/assets/a08747ce-d282-402c-a760-28195d3ee1ca" />
      <img width="1395" height="1389" alt="image" src="https://github.com/user-attachments/assets/8e38695e-e1da-4bfe-a90a-b5b8be9cf30b" />
      <img width="1402" height="1393" alt="image" src="https://github.com/user-attachments/assets/5bed3ae9-cea7-46f4-9acb-4fe0f75bf2d0" />

    - **Orientation**: after the rotation the recomputed angle is close to 0° (below 1.5°), which confirms the alignment. Nearly round objects have a poorly defined main axis.

      <img width="1398" height="1389" alt="image" src="https://github.com/user-attachments/assets/828f5d63-1a0d-451d-ad75-a0ab3c78dcec" />

    - **Shape comparison** (means for objects larger than 200 px):

      | Objects | Circularity | Rectangularity | Aspect ratio |
      |---|---|---|---|
      | wheat grains (compact) | 0.73 | 0.74 | 1.87 |
      | rice grains (elongated) | 0.50 | 0.76 | 2.94 |
      | rice clusters (irregular) | 0.19 | 0.40 | 1.82 |

      Circularity separates compact, elongated and irregular objects; rectangularity drops strongly for clusters; the aspect ratio measures elongation best. The descriptors can tell a cluster from a single object even when their areas are similar.

      <img width="1396" height="1390" alt="image" src="https://github.com/user-attachments/assets/c37635cd-950d-4c03-a96a-1d0d6bcb97cf" />

    - **Limits**: the contour is built from pixel steps (1 and √2), so even a digital disc has circularity ≈ 0.91, and objects of a few pixels get meaningless values (circularity > 1).
