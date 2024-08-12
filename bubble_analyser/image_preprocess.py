import cv2
import numpy as np
from skimage import (
    color,
    io,
    transform,
)


def load_image(image_path: str) -> np.ndarray:
    """Read and preprocess the input image.

    Args:
    image_path (str): The file path of the image to load.

    Returns:
    np.ndarray: The image read in ndarray format.
    """
    # Read the input image

    img = io.imread(image_path)

    return img


def get_greyscale(image: np.ndarray) -> np.ndarray:
    """Converts an image to grayscale if it is in RGB format.

    Args:
        image (np.ndarray): The input image to be converted.

    Returns:
        np.ndarray: The grayscale image.
    """
    if image.ndim > 2:
        image = color.rgb2gray(image)  # Convert to grayscale if the image is in RGB
    return image


def get_RGB(image: np.ndarray) -> np.ndarray:
    """Converts an image from BGR color space to RGB color space.

    Args:
    image (np.ndarray): The input image in BGR format.

    Returns:
    np.ndarray: The converted image in RGB format.
    """
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imgRGB


def resize_for_RGB(image: np.ndarray, img_resample_factor: float) -> np.ndarray:
    """Resizes an image in RGB format based on the provided resampling factor using method
    provided by openCV.

    Args:
        image (np.ndarray): The input image in RGB format.
        img_resample_factor (float): The factor by which the image will be resampled.

    Returns:
        np.ndarray: The resized image in RGB format.
    """
    scale_percent = img_resample_factor * 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    img_resample_dimension = (width, height)

    image = cv2.resize(image, img_resample_dimension, interpolation=cv2.INTER_AREA)
    return image


def resize_for_original_image(
    image: np.ndarray, img_resample_factor: float
) -> np.ndarray:
    """Resizes an image based on the provided resampling factor for original image using method
    provided by skimage.

    Args:
        image (np.ndarray): The input image to be resized.
        img_resample_factor (float): The factor by which the image will be resampled.

    Returns:
        np.ndarray: The resized image.
    """
    image = transform.resize(
        image,
        (
            int(image.shape[0] * img_resample_factor),
            int(image.shape[1] * img_resample_factor),
        ),
        anti_aliasing=True,
    )
    return image


def image_preprocess(
    img_path: str, img_resample: float
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses an image by loading it, converting it to 1. RGB, 2. grayscale
    and resizing it based on a provided resampling factor.
    The resized grayscale image (img) is for use in the following "default" watershed algorithm
    as a target image. And the RGB image (imgRGB) is for the visualization of the results.
    They are resized in different ways for different use cases (the output format of the
    methods are differnt).

    Args:
        img_path (str): The file path of the image to preprocess.
        img_resample (float): The resampling factor to apply to the image.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the resized grayscale image
        and the resized RGB image.
    """
    image = load_image(img_path)
    image_RGB = get_RGB(image)

    image = resize_for_original_image(image, img_resample)
    image = get_greyscale(image)

    image_RGB = resize_for_RGB(image_RGB, img_resample)

    return image, image_RGB
