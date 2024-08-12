import numpy as np
from background_subtraction_threshold import background_subtraction_threshold
from numpy.typing import NDArray
from skimage import (
    filters,
)


def otsu_threshold(target_img: np.ndarray) -> np.ndarray:
    """Apply Otsu's thresholding to the target image and return an inverted binary mask.

    This function takes a target image and a background image. It applies Otsu's thresholding
    to the target image to create a binary mask where the foreground objects are separated
    from the background. The binary mask is then inverted, so the foreground becomes the
    background and vice versa, the inversion is for easier processing in following morphologcal process.

    Args:
        target_img: A grayscale image (2D array) representing the target image.
        bknd_img: A grayscale image (2D array) representing the background image (not
        used in this function).

    Returns:
        A binary (inverted) image where the foreground and background are swapped.
    """
    binary_image: NDArray[np.bool_] = target_img > filters.threshold_otsu(
        target_img
    )  # Binary image using Otsu's thresholding
    return ~binary_image  # Invert the binary image


def select_threshold_method() -> str:
    """Selects a threshold method from a list of available options.
    Prompts the user to choose a threshold method from the list of options.
    The options are displayed with their corresponding numbers, and the user
    is asked to input the number of their chosen method.

    Returns:
        str: The chosen threshold method.
    """
    options = ["OTSU's method", "Background subtraction"]
    print("Select a threshold method:")
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    choice = input("Enter the number of your choice: ")
    return options[int(choice) - 1]


def threshold(
    target_img: np.ndarray, bknd_img: np.ndarray, threshold_value: float
) -> np.ndarray:
    """Applies a threshold to the target image based on the selected method.

    Args:
        target_img (np.ndarray): The target image to apply the threshold to.
        bknd_img (np.ndarray): The background image used for background subtraction.
        threshold_value (float): The threshold value used for background subtraction.

    Returns:
        np.ndarray: The thresholded image.
    """
    method = select_threshold_method()
    if method == "OTSU's method":
        return otsu_threshold(target_img)
    elif method == "Background subtraction":
        return background_subtraction_threshold(target_img, bknd_img, threshold_value)
    else:
        raise ValueError(f"Unsupported threshold method: {method}")
