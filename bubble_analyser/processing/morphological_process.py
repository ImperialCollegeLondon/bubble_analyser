"""Morphological Processing Function for filling holes and clear borders.

This module includes functions for advanced image processing using morphological
operations tailored to for filling holes and clear borders for further analysis. It
specifically focuses on refining the binary masks generated during image segmentation
processes.

Function:
- morphological_process(target_img, element_size): Enhances a binary image by
  applying morphological operations such as closing, hole filling, and border clearing.

This function is particularly useful in contexts where binary images derived from
thresholding or other segmentation methods contain noise, small holes, or artifacts
that can interfere with further analysis. By using operations like closing to connect
nearby regions, filling holes to ensure that objects are solid, and clearing borders
to remove partial objects, this function prepares images for more reliable and robust
analysis.
"""

import time
from pathlib import Path

# import matplotlib
# matplotlib.use('TkAgg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt
from scipy import ndimage
from skimage import (
    morphology,
    segmentation,
)


def morphological_process(target_img: npt.NDArray[np.bool_], element_size: int = 8) -> npt.NDArray[np.int_]:
    """Apply morphological operations to process the target image.

    This function performs a series of morphological operations on the input image,
    including closing, filling holes, and clearing borders. These operations help in
    refining the binary image by removing noise and filling gaps.

    Args:
        target_img: A binary image (numpy array) where the regions of interest are
        typically in white (True) and the background in black (False).
        element_size: A structuring element used for morphological closing, typically a
        disk-shaped array.


    Returns:
        A processed binary image (numpy array) where the regions of interest are more
        defined, with filled holes and cleared borders.
    """
    start_time = time.perf_counter()
    # element_size = morphology.disk(element_size)

    kernel = morphology.disk(element_size).astype(np.uint8)
    dilated = cv2.dilate(target_img.astype(np.uint8), kernel, iterations=1)
    image_processed_closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # image_processed_closed = cv2.morphologyEx(
    #     target_img.astype(np.uint8), cv2.MORPH_CLOSE, element_size
    # )  # type: ignore

    print("Time consumed for closing: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    image_processed_filled = ndimage.binary_fill_holes(image_processed_closed)
    print("Time consumed for filling holes: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    image_processed_cleared = segmentation.clear_border(image_processed_filled)
    print("Time consumed for clearing borders: ", time.perf_counter() - start_time)

    image_processed_cleared = image_processed_cleared.astype(np.uint8)
    # opening = cv2.morphologyEx(B,cv2.MORPH_OPEN,kernel, iterations = 2)

    plt.figure()
    plt.subplot(231)
    plt.title("1. Original image")
    plt.imshow(target_img, cmap="gray")
    plt.subplot(232)
    plt.title("2. close")
    plt.imshow(image_processed_closed * 255, cmap="gray")
    plt.subplot(233)
    plt.title("3. fill holes")
    plt.imshow(image_processed_filled * 255, cmap="gray")
    plt.subplot(234)
    plt.title("4. clear border")
    plt.imshow(image_processed_cleared, cmap="gray")
    plt.show()

    return image_processed_cleared


if __name__ == "__main__":
    # Define the image path
    img_path = Path("../../tests/test_image_thresholded_otsu.JPG")
    output_path = Path("../../tests/test_image_mt.JPG")

    # Load the image
    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    img_processed: npt.NDArray[np.int_]
    # Apply morphological process to the grayscale image
    img_processed = morphological_process(img_binary, 8)  # type: ignore

    # Save the processed image
    cv2.imwrite(str(output_path), img_processed.astype(np.uint8) * 255)

    print(f"Morphological processed image saved to: {output_path}")
