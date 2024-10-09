"""Main Thresholding Functions.

This module provides functions for applying various thresholding techniques to images,
aimed at segmenting objects from their backgrounds. It includes implementations of
Otsu's method and a customizable thresholding approach that involves background
subtraction.

Functions:
- otsu_threshold(target_img): Applies Otsu's method to a grayscale image to create
  a binary mask, which is then inverted for consistency with other processing steps.
- select_threshold_method(): Provides an interactive prompt for the user to select
  a thresholding method from available options, enhancing flexibility in choosing
  the appropriate method for different scenarios.
- threshold(target_img, bknd_img, threshold_value): Applies the chosen threshold method
  to the target image, supporting either Otsu's method or a custom threshold based on
  background subtraction, determined by user input.

These thresholding functions are adaptable to various use cases, from basic academic
projects to complex industrial applications requiring robust foreground-background
segmentation. They are particularly useful in workflows that require pre-processing
before detailed image analysis, such as feature detection or object classification.

Usage:
The functions can be directly called with appropriate parameters, with `threshold`
function allowing for dynamic method selection based on runtime decisions. This
design ensures that users can select the most suitable thresholding technique
based on the specific characteristics of the images they are working with.
"""

import numpy as np
from numpy import typing as npt
from skimage import (
    filters,
)

from .background_subtraction_threshold import background_subtraction_threshold


def otsu_threshold(target_img: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
    """Apply Otsu's thresholding to the target image and return an inverted binary mask.

    This function takes a target image and a background image. It applies Otsu's
    thresholding to the target image to create a binary mask where the foreground
    objects are separatedfrom the background. The binary mask is then inverted, so
    the foreground becomes the background and vice versa, the inversion is for easier
    processing in following morphologcal process.

    Args:
        target_img: A grayscale image (2D array) representing the target image.
        bknd_img: A grayscale image (2D array) representing the background image (not
        used in this function).

    Returns:
        A binary (inverted) image where the foreground and background are swapped.
    """
    binary_image = target_img > filters.threshold_otsu(
        target_img
    )  # Binary image using Otsu's thresholding

    return binary_image

def threshold(
    target_img: npt.NDArray[np.int_],
    bknd_img: npt.NDArray[np.int_],
    threshold_value: float,
) -> npt.NDArray[np.bool]:
    """Applies a threshold to the target image based on the selected method.

    Args:
        target_img (npt.NDArray): The target image to apply the threshold to.
        bknd_img (npt.NDArray): The background image used for background subtraction.
        threshold_value (float): The threshold value used for background subtraction.

    Returns:
        npt.NDArray: The thresholded image.
    """
    target_img = background_subtraction_threshold(target_img, bknd_img)
    return otsu_threshold(target_img)


def threshold_without_background(
    target_img: npt.NDArray[np.int_], threshold_value: float
) -> npt.NDArray[np.bool_]:
    """Applies a threshold to the target image without background subtraction.

    Args:
        target_img (npt.NDArray): The target image to apply the threshold to.
        threshold_value (float): The threshold value used for background subtraction.

    Returns:
        npt.NDArray: The thresholded image.
    """
    return ~otsu_threshold(target_img)
