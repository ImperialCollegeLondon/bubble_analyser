import cv2
import numpy as np


def convert_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts an image to grayscale.

    Args:
        image (np.ndarray): The input image to be converted.

    Returns:
        np.ndarray: The converted image in grayscale.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def background_subtraction(
    target_img: np.ndarray, background_img: np.ndarray
) -> np.ndarray:
    """Performs background subtraction on two images.

    Args:
        target_img (np.ndarray): The target image where the objects of interest are located.
        background_img (np.ndarray): The background image (without objects of interest).

    Returns:
        np.ndarray: The difference image after background subtraction.
    """
    difference_img = cv2.absdiff(target_img, background_img)
    return difference_img


def threshold(difference_img: np.ndarray, threshold_value: int) -> np.ndarray:
    """Applies a binary threshold to the given difference image.

    Args:
        difference_img (np.ndarray): The input difference image to be thresholded.
        threshold_value (int): The threshold value to apply to the difference image.

    Returns:
        np.ndarray: The thresholded image.
    """
    _, thresholded_img = cv2.threshold(
        difference_img, threshold_value, 255, cv2.THRESH_BINARY
    )
    return thresholded_img


def background_subtraction_threshold(
    target_img: np.ndarray, background_img: np.ndarray, threshold_value: int
) -> np.ndarray:
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
