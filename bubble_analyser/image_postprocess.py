"""Functions that process the image after being watershed segmented.

This module currently provides a single function, overlay_labels_on_rgb, which takes an
RGB image and a 2D array of labeled regions and combines them into a single image with
the labeled regions overlaid on the original image. The labeled regions are represented
with a unique color for each label, and the transparency of the overlay can be
controlled using the 'alpha' parameter.

The function returns the resulting image as a 3D array in float format with range[0, 1].
"""

import cv2
import numpy as np
from numpy import typing as npt


def overlay_labels_on_rgb(
    imgRGB: npt.NDArray[np.int_], labels: npt.NDArray[np.int_], alpha: float = 0.5
) -> npt.NDArray[np.int_]:
    """Overlay labeled regions on an RGB image with a transparent color.

    Parameters
    ----------
    imgRGB : ndarray
        The RGB image to overlay the labeled regions on.
    labels : ndarray
        A 2D array of labeled regions, where each unique label is represented by a
        distinct integer.
    alpha : float, optional
        The transparency of the overlay, with 0 being fully transparent and 1 being
        fully opaque. Default is 0.5.

    Returns:
    -------
    ndarray
        The resulting image with the labeled regions overlaid on the original image.
    """
    # Ensure imgRGB is in uint8 format
    imgRGB = (imgRGB * 255.0).astype(np.uint8) if imgRGB.max() <= 1 else imgRGB

    unique_labels = np.unique(labels)

    # Convert the label image to BGR (OpenCV's color format is BGR, not RGB)
    colored_labels = np.zeros_like(imgRGB)

    for label in unique_labels:
        if label == 1:  # Skip the background (assuming label 0 is background)
            continue
        # Create a mask for the current label
        # Generate random hue (0-179 in OpenCV's HSV), max saturation, and max brightness
        hue = np.random.randint(0, 179)
        saturation = 255  # Max saturation
        value = 255  # Max brightness
        color_hsv = np.array(
            [[[hue, saturation, value]]], dtype=np.uint8
        )  # HSV color format
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][
            0
        ]  # Convert HSV to BGR color

        # Create a mask for the current label
        mask = labels == label
        colored_labels[mask] = color_bgr

    # Blend the colored labels with the original image using transparency (alpha)
    label_overlay = cv2.addWeighted(imgRGB, 1 - alpha, colored_labels, alpha, 0)

    return label_overlay  # type: ignore
