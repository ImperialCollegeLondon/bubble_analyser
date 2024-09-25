"""
Functions that process the image after being watershed segmented.

This module currently provides a single function, overlay_labels_on_rgb, which takes an 
RGB image and a 2D array of labeled regions and combines them into a single image with
the labeled regions overlaid on the original image. The labeled regions are represented
with a unique color for each label, and the transparency of the overlay can be 
controlled using the 'alpha' parameter.

The function returns the resulting image as a 3D array in float format with range [0, 1].
"""

import numpy as np
from skimage.color import label2rgb
import matplotlib.pyplot as plt

def overlay_labels_on_rgb(
    imgRGB: np.ndarray, 
    labels: np.ndarray, 
    alpha: float = 0.5
) -> np.ndarray:
    """Overlays labeled regions on the original RGB image with transparency.

    Args:
        imgRGB: Original RGB image (3D array).
        labels: Labeled regions (2D array).
        alpha: The transparency factor for the label overlay (0.0 is fully transparent, 1.0 is fully opaque).

    Returns:
        The RGB image with labeled regions overlaid.
    """

    # Ensure imgRGB is in float format with range [0, 1]
    imgRGB = imgRGB.astype(np.float32) / 255.0 if imgRGB.max() > 1 else imgRGB

    # Apply skimage's label2rgb function to map each label to a unique color
    label_overlay = label2rgb(labels, image=imgRGB, bg_label = 1, alpha=alpha)

    return label_overlay

