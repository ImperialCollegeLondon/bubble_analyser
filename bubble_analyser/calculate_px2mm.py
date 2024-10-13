"""Bubble Analyser: Image Processing for Circular Feature Detection.

This module provides a suite of tools for image manipulation and measurement calibration
using computer vision techniques. It includes functions to resize images, draw on images
interactively, and calculate real-world measurements from pixels.

The functions in this module utilize OpenCV and NumPy to perform tasks such as image
resizing, interactive line drawing for measurement marking, pixel distance calculations,
and conversionfrom pixel measurements to real-world units (e.g., centimeters). These
capabilities are particularly useful in applications where precision in spatial
measurements is required, such as in quality control, materials science, and medical
imaging.

Key Functions:
- resize_to_target_width(image, target_width): Resizes an image to a specified target
  width while maintaining the aspect ratio.
- draw_line(event, x, y, flags, param): A callback function that allows interactive line
  drawing on an image displayed in an OpenCV window.
- get_pixel_distance(img): Displays an image and allows the user to draw a line, then
  calculates the pixel distance between the endpoints of the line.
- get_cm_per_pixel(pixel_distance, scale_percent, img_resample): Calculates the
  conversion ratio from pixels to centimeters, taking into account any image resizing
  that has been applied.
- calculate_px2cm(image_path, img_resample): Orchestrates the process of loading an
  image, resizing it, allowing the user to mark a measurement, and calculating a
  pixel-to-centimeter conversion factor.

Each function is designed to be modular, allowing for flexible integration into broader
image processing and analysis workflows. The module facilitates the extraction of
quantitative data from images, which can be critical for applications requiring detailed
spatial analysis.
"""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from .image_preprocess import load_image


def resize_to_target_width(
    image: npt.NDArray[np.int_], target_width: int = 1000
) -> tuple[npt.NDArray[np.int_], float]:
    """Resizes an image to a specified target width while maintaining the aspect ratio.

    Args:
        image (npt.npt.NDArray[np.int_]): The input image to be resized.
        target_width (int, optional): The desired width of the resized image. Defaults
        to 1000.

    Returns:
        npt.npt.NDArray[np.int_]: The resized image.
        scale_percent: The scaling percentage applied to the image during resizing.
    """
    # Scale down the image to a width of 1000 pixels, keeping the aspect ratio the same
    scale_percent: float = (
        target_width / image.shape[1]
    )  # Calculate the scale percent to make width 1000
    width: int = target_width  # Set the new width to 1000 pixels
    height: int = int(
        image.shape[0] * scale_percent
    )  # Adjust the height to maintain the aspect ratio
    dim: tuple[int, int] = (width, height)  # Define the new dimensions
    image_resized = cv2.resize(
        image, dim, interpolation=cv2.INTER_AREA
    )  # Resize the image

    return cast(npt.NDArray[np.int_], image_resized), scale_percent


def draw_line(event: int, x: int, y: int, flags: int, param: dict[str, object]) -> None:
    """Callback function to draw a line on the image.

    This function is used as a mouse callback to allow the user to draw a line on the
    image.

    Args:
        event: The type of mouse event (e.g., left button down, mouse move, left button
        up).
        x: The x-coordinate of the mouse event.
        y: The y-coordinate of the mouse event.
        flags: Any relevant flags passed by OpenCV.
        param: A dictionary containing reference points and drawing state.
    """
    refPt = cast(list[tuple[int, int]], param["refPt"])
    img = cast(npt.NDArray[np.uint8], param["img"])
    img_copy = cast(npt.NDArray[np.uint8], param["img_copy"])

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        param["drawing"] = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if param["drawing"]:
            img_copy[:] = img[:]  # Reset to the original image before drawing the line
            cv2.line(img_copy, refPt[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        param["drawing"] = False
        cv2.line(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", img)


def get_pixel_distance(img: npt.NDArray[np.int_]) -> float:
    """Display the image and allow the user to draw a line representing 1 cm.

    This function uses OpenCV to display the image and capture the user input
    for drawing a line that represents 1 cm on the ruler. It calculates the
    Euclidean distance between the two points of the line in pixels.

    Args:
        img: The image on which the user will draw a line.

    Returns:
        The distance in pixels between the two drawn points. Returns 0 if the
        line was not drawn correctly.
    """
    refPt: list[tuple[int, int]] = []
    drawing: bool = False
    img_copy: npt.NDArray[np.int_] = img.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback(
        "image",
        draw_line,  # type: ignore
        {"refPt": refPt, "drawing": drawing, "img": img, "img_copy": img_copy},
    )

    print(
        "Draw a line representing 1 cm according to the ruler's scaling in the image."
    )

    while True:
        cv2.imshow("image", img_copy)
        key: int = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(refPt) == 2:
        pixel_distance: float = np.sqrt(
            (refPt[1][0] - refPt[0][0]) ** 2 + (refPt[1][1] - refPt[0][1]) ** 2
        )
        return pixel_distance
    else:
        print("Line was not drawn correctly.")
        return 0.0


def get_mm_per_pixel(
    pixel_distance: float, scale_percent: float, img_resample: float
) -> float:
    """Calculate the conversion ratio from pixels to centimeters.

    Args:
        pixel_distance: The distance in pixels between the two drawn points.
        scale_percent: The scaling percentage applied to the image during resizing.
        img_resample: The resampling factor applied to the original target and
        background image.

    Returns:
        The conversion factor in centimeters per pixel, corrected for the resampling
        applied to the original image.
    """
    original_pixel_distance: float = pixel_distance / scale_percent
    mm_per_pixel: float = 10.0 / original_pixel_distance
    mm_per_pixel = mm_per_pixel / img_resample
    return mm_per_pixel


def calculate_px2mm(image_path: Path, img_resample: float) -> tuple[float, float]:
    """Calculates the conversion factor from pixels to centimeters.

    This function reads an image of a ruler, allows the user to draw a line
    correspondingnto 1 cm on the ruler, and calculates the pixel-to-centimeter
    conversion factor. The image is scaled down for easier interaction, but the final
    calculation accounts forthis scaling as well as the resample factor for target and
    background images to ensure accuracy relative to the original image size.

    Args:
        image_path (str): The path to the image file.
        img_resample (float): The resampling factor applied to the original target and
        background image.

    Returns:
        float: The conversion factor in centimeters per pixel, corrected for the
        resampling applied to the original image.
    """
    image = load_image(image_path)
    image, scale_percent = resize_to_target_width(image)
    pixel_distance: float = get_pixel_distance(image)
    if pixel_distance > 0:
        mm_per_pixel: float = get_mm_per_pixel(
            pixel_distance, scale_percent, img_resample
        )
        print(f"Conversion factor: {mm_per_pixel} mm per pixel")
        pixel_per_mm = 1 / mm_per_pixel
        print(f"Conversion factor: {pixel_per_mm} pixels per mm")
    return mm_per_pixel, pixel_per_mm
