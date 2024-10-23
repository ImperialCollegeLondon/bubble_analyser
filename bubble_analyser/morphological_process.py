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

import numpy as np
import time
import cv2
from numpy import typing as npt
from scipy import ndimage
from skimage import (
    morphology,
    segmentation,
)


def morphological_process(
    target_img: npt.NDArray[np.bool_], element_size: int
) -> npt.NDArray[np.int_]:
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
    

    # image_processed = morphology.closing(target_img, element_size)
    image_processed = cv2.morphologyEx(target_img.astype(np.uint8), cv2.MORPH_CLOSE, element_size)
    print("Time consumed for closing: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    image_processed = ndimage.binary_fill_holes(image_processed)
    print("Time consumed for filling holes: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    image_processed = segmentation.clear_border(image_processed)
    print("Time consumed for clearing borders: ", time.perf_counter() - start_time)

    image_processed = image_processed.astype(np.uint8)
    # opening = cv2.morphologyEx(B,cv2.MORPH_OPEN,kernel, iterations = 2)

    return image_processed
