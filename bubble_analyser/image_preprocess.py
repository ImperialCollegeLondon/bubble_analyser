"""Image Preprocessing Functions.

This module provides a collection of functions for image loading, color space conversion
, and resizing. It supports the preprocessing steps required for image analysis tasks,
especially in contexts where images need to be adapted for algorithmic processing and
visualization.

Key Functions:
- load_image(image_path): Loads an image from a specified path into a NumPy array.
- get_greyscale(image): Converts an RGB image to grayscale, facilitating algorithms
  that require single-channel input.
- get_RGB(image): Converts an image from BGR (common in OpenCV) to RGB format, suitable
  for consistent image display and processing.
- resize_for_RGB(image, img_resample_factor): Resizes an RGB image according to a
  specified resampling factor, typically used to reduce the image size for faster
  processing without losing significant detail.
- resize_for_original_image(image, img_resample_factor): Similar to resize_for_RGB but
  uses skimage's transform for resizing, providing a high-quality downsampling suitable
  for analytical purposes.
- image_preprocess(img_path, img_resample): Orchestrates the loading, converting, and
  resizing of an image. It outputs both a grayscale version for processing and an RGB
  version for visualization.

These functions are designed to be modular and can be combined in different ways
depending on the specific requirements of the image processing task at hand. For example
, in a typical workflow for image analysis, an image might be loaded, converted to
grayscale for analysis, and also kept in RGB for result visualization.

Usage:
These utilities are particularly useful in applications like computer vision and digital
image processing where preprocessing steps are crucial for subsequent analysis, such as
object detection, pattern recognition, and more.
"""

import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy import typing as npt
from skimage import (
    color,
    io,
)


def load_image(image_path: Path) -> npt.NDArray[np.int_]:
    """Read and preprocess the input image.

    Args:
    image_path (str): The file path of the image to load.

    Returns:
    npt.NDArray: The image read in ndarray format.
    """
    # Read the input image

    img = io.imread(image_path)

    return img


def get_greyscale(image: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Converts an image to grayscale if it is in RGB format.

    Args:
        image (npt.NDArray): The input image to be converted.

    Returns:
        npt.NDArray: The grayscale image.
    """
    if image.ndim > 2:
        image = color.rgb2gray(image)  # Convert to grayscale if the image is in RGB
    return image


def get_RGB(image: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Converts an image from BGR color space to RGB color space.

    Args:
    image (npt.NDArray): The input image in BGR format.

    Returns:
    npt.NDArray: The converted image in RGB format.
    """
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cast(npt.NDArray[np.int_], imgRGB)


def resize_for_RGB(
    image: npt.NDArray[np.int_], img_resample_factor: float
) -> npt.NDArray[np.int_]:
    """Resizes an image in RGB format based on the provided resampling factor.

    Args:
        image (npt.NDArray): The input image in RGB format.
        img_resample_factor (float): The factor by which the image will be resampled.

    Returns:
        npt.NDArray: The resized image in RGB format.
    """
    scale_percent = img_resample_factor * 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    img_resample_dimension = (width, height)

    image_resized = cv2.resize(
        image, img_resample_dimension, interpolation=cv2.INTER_AREA
    )
    return cast(npt.NDArray[np.int_], image_resized)


def resize_for_original_image(
    image: npt.NDArray[np.int_], img_resample_factor: float
) -> npt.NDArray[np.int_]:
    """Resizes an image based on the provided resampling factor for original image.

    Args:
        image (npt.NDArray): The input image to be resized.
        img_resample_factor (float): The factor by which the image will be resampled.

    Returns:
        npt.NDArray: The resized image.
    """
    # image = transform.resize(
    #     image,
    #     (
    #         int(image.shape[0] * img_resample_factor),
    #         int(image.shape[1] * img_resample_factor),
    #     ),
    #     anti_aliasing=True,
    # )
    # return image
    return cv2.resize(
        image,
        (0, 0),
        fx=img_resample_factor,
        fy=img_resample_factor,
        interpolation=cv2.INTER_AREA,
    )


def image_preprocess(
    img_path: Path, img_resample: float
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Load an image, resizing it based on a provided resampling factor.

    The resized grayscale image (img) is for use in the following "default" watershed
    algorithm as a target image. And the RGB image (imgRGB) is for the visualization of
    the results. They are resized in different ways for different use cases (the output
    format of the methods are differnt).

    Args:
        img_path (str): The file path of the image to preprocess.
        img_resample (float): The resampling factor to apply to the image.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: A tuple containing the resized grayscale image
        and the resized RGB image.
    """
    start_time = time.perf_counter()
    image = load_image(img_path)
    print("Time used for load_image: ", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    image_RGB = get_RGB(image)
    print("Time used for get_RGB: ", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    image = resize_for_original_image(image, img_resample)
    print("Time used for resize_for_original_image: ", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    image = get_greyscale(image)
    print("Time used for get_greyscale: ", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    image_RGB = resize_for_RGB(image_RGB, img_resample)
    print("Time used for resize_for_RGB: ", time.perf_counter() - start_time)

    return image, image_RGB
