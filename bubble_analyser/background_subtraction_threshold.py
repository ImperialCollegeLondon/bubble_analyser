"""Background Subtraction and Thresholding: Isolate objects from backgrounds.

This module provides tools for image preprocessing including grayscale conversion,
background subtraction, and thresholding. It is designed to handle images where objects
of interest need to be isolated from their backgrounds for further analysis.

Functions:
    convert_grayscale(image): Converts a color image to grayscale.

    background_subtraction(target_img, background_img): Subtracts the background image
    from the target image to highlight differences.

    threshold(difference_img, threshold_value): Applies a binary threshold to an image
    to create a binary mask.

    background_subtraction_threshold(target_img, background_img, threshold_value):
    Combines background subtraction and thresholding to isolate objects of interest in
    an image.

These functions are used to preprocess images for applications such as object detection,
where isolating the changes between images or from a background is necessary. Each
function is designed to be modular, allowing them to be used independently or in
sequence depending on the requirements of the task.
"""

from typing import cast

import cv2
import numpy as np
from numpy import typing as npt


def convert_grayscale(image: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Converts an image to grayscale.

    Args:
        image (npt.NDArray): The input image to be converted.

    Returns:
        npt.NDArray: The converted image in grayscale.
    """
    if len(image.shape) == 3:
        image = cast(npt.NDArray[np.int_], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return image


def background_subtraction(
    target_img: npt.NDArray[np.int_], background_img: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """Performs background subtraction on two images.

    Args:
        target_img (npt.NDArray): The target image where the objects of interest are
        located.
        background_img (npt.NDArray): The background image (without object of interest).

    Returns:
        npt.NDArray: The difference image after background subtraction.
    """
    difference_img = cv2.absdiff(target_img, background_img)
    return cast(npt.NDArray[np.int_], difference_img)


def threshold(
    difference_img: npt.NDArray[np.int_], threshold_value: float
) -> npt.NDArray[np.bool_]:
    """Applies a binary threshold to the given difference image.

    Args:
        difference_img (npt.NDArray): The input difference image to be thresholded.
        threshold_value (int): The threshold value to apply to the difference image.

    Returns:
        npt.NDArray: The thresholded image.
    """
    _, thresholded_img = cv2.threshold(
        difference_img, threshold_value, 255, cv2.THRESH_BINARY
    )
    return cast(npt.NDArray[np.bool_], thresholded_img)


def background_subtraction_threshold(
    target_img: npt.NDArray[np.int_],
    background_img: npt.NDArray[np.int_],
    threshold_value: float,
) -> npt.NDArray[np.bool_]:
    """Perform background subtraction and apply thresholding.

    Args:
        target_img: The target image where the objects of interest are located.
        background_img: The background image (without objects of interest).
        threshold_value: The threshold value to apply after background subtraction.

    Returns:
        A binary image with the objects of interest isolated.
    """
    # Ensure both images are in grayscale
    target_img = convert_grayscale(target_img)
    background_img = convert_grayscale(background_img)

    # Subtract the background image from the target image
    difference_img = background_subtraction(target_img, background_img)

    # Apply a threshold to the difference image
    thresholded_img = threshold(difference_img, threshold_value)

    return thresholded_img
