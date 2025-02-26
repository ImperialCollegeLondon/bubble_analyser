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

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy import typing as npt
from skimage import (
    filters,
)


class ThresholdMethods:
    def otsu_threshold(self, target_img: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
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

    def convert_grayscale(self, image: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
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
        self, target_img: npt.NDArray[np.int_], background_img: npt.NDArray[np.int_]
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

    def threshold_with_background(
        self, target_img: npt.NDArray[np.int_], bknd_img: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.bool]:
        """Applies a threshold to the target image based on the selected method.

        Args:
            target_img (npt.NDArray): The target image to apply the threshold to.
            bknd_img (npt.NDArray): The background image used for background subtraction.
            threshold_value (float): The threshold value used for background subtraction.

        Returns:
            npt.NDArray: The thresholded image.
        """
        # Ensure both images are in grayscale
        target_img = self.convert_grayscale(target_img)
        background_img = self.convert_grayscale(bknd_img)

        # Subtract the background image from the target image
        difference_img = self.background_subtraction(target_img, background_img)

        return self.otsu_threshold(difference_img)

    def threshold_without_background(
        self, target_img: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.bool_]:
        """Applies a threshold to the target image without background subtraction.

        Args:
            target_img (npt.NDArray): The target image to apply the threshold to.
            threshold_value (float): The threshold value used for background subtraction.

        Returns:
            npt.NDArray: The thresholded image.
        """
        return ~self.otsu_threshold(target_img)


if __name__ == "__main__":
    # Define the image path
    img_path = Path("../../tests/test_image_grey.JPG")
    bknd_path = Path("../../tests/background_image_grey.JPG")
    img_thresholded_path_otsu = Path("../../tests/test_image_thresholded_otsu.JPG")
    img_thresholded_path_with_bknd = Path(
        "../../tests/test_image_thresholded_with_bknd.JPG"
    )

    # Create an instance of the ThresholdMethods class
    threshold_methods = ThresholdMethods()

    # Load the image
    img: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], cv2.imread(str(img_path)))
    bknd: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], cv2.imread(str(bknd_path)))

    # Convert the image to grayscale
    img_grey: npt.NDArray[np.int_] = threshold_methods.convert_grayscale(img)
    bknd_grey: npt.NDArray[np.int_] = threshold_methods.convert_grayscale(bknd)

    # Apply Otsu's thresholding to the grayscale image
    img_thresholded_otsu = threshold_methods.threshold_without_background(img_grey)
    img_thresholded_with_bknd = threshold_methods.threshold_with_background(
        img_grey, bknd_grey
    )

    # Save the thresholded image
    cv2.imwrite(
        str(img_thresholded_path_otsu), img_thresholded_otsu.astype(np.uint8) * 255
    )
    cv2.imwrite(
        str(img_thresholded_path_with_bknd),
        img_thresholded_with_bknd.astype(np.uint8) * 255,
    )
    print(f"Thresholded image saved to: {img_thresholded_path_otsu}")
