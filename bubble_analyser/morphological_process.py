import numpy as np
from scipy import ndimage
from skimage import (
    morphology,
    segmentation,
)


def morphological_process(target_img: np.ndarray, element_size: int) -> np.ndarray:
    """Apply morphological operations to process the target image.

    This function performs a series of morphological operations on the input image,
    including closing, filling holes, and clearing borders. These operations help in
    refining the binary image by removing noise and filling gaps.

    Args:
        target_img: A binary image (numpy array) where the regions of interest are
        typically in white (True) and the background in black (False).
        se: A structuring element used for morphological closing, typically a disk-shaped array.


    Returns:
        A processed binary image (numpy array) where the regions of interest are more
        defined, with filled holes and cleared borders.
    """
    # Perform morphological closing and fill holes
    image_processed = morphology.closing(target_img, element_size)
    image_processed = ndimage.binary_fill_holes(image_processed)
    image_processed = segmentation.clear_border(image_processed)

    image_processed = image_processed.astype(np.uint8)
    # opening = cv2.morphologyEx(B,cv2.MORPH_OPEN,kernel, iterations = 2)

    return image_processed
