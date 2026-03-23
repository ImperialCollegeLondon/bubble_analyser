"""Image Preprocessing Functions.

This module provides a collection of functions for image loading, color space conversion
, and resizing. It supports the preprocessing steps required for image analysis tasks,
especially in contexts where images need to be adapted for algorithmic processing and
visualization.
"""

import logging
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


def resize_for_RGB(image: npt.NDArray[np.int_], resample: float) -> npt.NDArray[np.int_]:
    """Resizes an image in RGB format to a target width while maintaining aspect ratio.

    Args:
        image (npt.NDArray): The input image in RGB format.
        resample (float): The resample factor for the resized image.

    Returns:
        npt.NDArray: The resized image in RGB format.
    """
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width
    target_width = int(original_width * resample)
    target_height = int(target_width * aspect_ratio)

    image_resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return cast(npt.NDArray[np.int_], image_resized)


def resize_for_original_image(image: npt.NDArray[np.int_], resample: float) -> npt.NDArray[np.int_]:
    """Resizes an image to a target width while maintaining aspect ratio.

    Args:
        image (npt.NDArray): The input image to be resized.
        resample (float): The resample factor for the resized image.

    Returns:
        npt.NDArray: The resized image.
    """
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width
    target_width = int(original_width * resample)
    target_height = int(target_width * aspect_ratio)

    resize_image: npt.NDArray[np.int_] = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )  # type: ignore
    return resize_image


def image_preprocess(img_path: Path, resample: float) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Load an image, resizing it to a target width while maintaining aspect ratio.

    The resized grayscale image (img) is for use in the following "default" watershed
    algorithm as a target image. And the RGB image (imgRGB) is for the visualization of
    the results. They are resized in different ways for different use cases (the output
    format of the methods are differnt).

    Args:
        img_path (str): The file path of the image to preprocess.
        resample (float): The resample factor for the resized images.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: A tuple containing the resized grayscale image
        and the resized RGB image.
    """
    logging.info("Image preprocessing...")
    start_time = time.perf_counter()
    image = load_image(img_path)
    logging.info(f"Time used for load_image: {time.perf_counter() - start_time}")

    start_time = time.perf_counter()
    image_RGB = get_RGB(image)
    logging.info(f"Time used for get_RGB:  {time.perf_counter() - start_time}")

    start_time = time.perf_counter()
    image = resize_for_original_image(image, resample)
    logging.info(f"Time used for resize_for_original_image:  {time.perf_counter() - start_time}")

    start_time = time.perf_counter()
    image = get_greyscale(image)
    logging.info(f"Time used for get_greyscale:  {time.perf_counter() - start_time}")

    start_time = time.perf_counter()
    image_RGB = resize_for_RGB(image_RGB, resample)
    logging.info(f"Time used for resize_for_RGB:  {time.perf_counter() - start_time}")

    logging.info("Image preprocessing finished.")
    return image, image_RGB


if __name__ == "__main__":
    # Define the image path and target width
    # img_path = Path("../../datafile/calibration_files/Background.png")
    # output_path_grey = Path("../../tests/background_image_grey.JPG")
    # output_path_rgb = Path("../../tests/background_image_rgb.JPG")

    img_path = Path("../../tests/test_image_raw_30ppm.JPG")
    output_path_grey = Path("../../tests/test_image_grey.JPG")
    output_path_rgb = Path("../../tests/test_image_rgb.JPG")

    target_width = 800  # Target width in pixels

    img, img_rgb = image_preprocess(img_path, target_width)
    img_grey = get_greyscale(img)
    # Save the output images
    img_grey_path = Path(output_path_grey)
    img_rgb_path = Path(output_path_rgb)
    cv2.imwrite(str(img_grey_path), img_grey * 255)
    cv2.imwrite(str(img_rgb_path), img_rgb)

    print(f"Grayscale image saved to: {img_grey_path}")
    print(f"RGB image saved to: {img_rgb_path}")
