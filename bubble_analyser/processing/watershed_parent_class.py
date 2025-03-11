"""Watershed Segmentation Base Class.

This module provides a base class for watershed segmentation algorithms used in bubble analysis.
It implements common functionality shared by different watershed segmentation approaches,
including thresholding, morphological processing, distance transform, and watershed segmentation.

Classes:
    WatershedSegmentation: Base class for watershed segmentation implementations that provides
                          common methods and attributes used by derived classes.

The WatershedSegmentation class is designed to be extended by specific watershed algorithm
implementations, which can customize the segmentation process while reusing the core
functionality provided by this base class.
"""

from typing import cast

import cv2
import numpy as np
from numpy import typing as npt

from bubble_analyser.processing.image_postprocess import overlay_labels_on_rgb
from bubble_analyser.processing.morphological_process import morphological_process
from bubble_analyser.processing.threshold_methods import ThresholdMethods


class WatershedSegmentation:
    """Base class for watershed segmentation implementations.

    This class provides common functionality for watershed segmentation algorithms,
    including image preprocessing, thresholding, morphological operations, and the
    watershed transform itself. It is designed to be extended by specific watershed
    implementations that can customize the segmentation process.

    Attributes:
        img_grey (npt.NDArray[np.int_]): Input grayscale image.
        img_grey_thresholded (npt.NDArray[np.bool_]): Binary image after thresholding.
        img_grey_morph (npt.NDArray[np.int_]): Image after morphological operations.
        img_grey_dt (npt.NDArray[np.int_]): Distance transform of the binary image.
        img_rgb (npt.NDArray[np.int_]): Input RGB image for visualization.
        element_size (int): Size of structuring element for morphological operations.
        connectivity (int): Pixel connectivity for connected components labeling.
        labels (npt.NDArray[np.int_]): Connected component labels.
        labels_watershed (npt.NDArray[np.int_]): Final watershed segmentation labels.
        labels_on_img (npt.NDArray[np.int_]): Visualization of labels on RGB image.
        if_bknd_img (bool): Flag indicating if background image is used.
        bknd_img (npt.NDArray[np.int_]): Background image for subtraction if used.
    """

    def __init__(
        self,
        img_grey: npt.NDArray[np.int_],
        img_rgb: npt.NDArray[np.int_],
        element_size: int = 5,
        connectivity: int = 4,
        if_bknd_img: bool = False,
        bknd_img: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], None),
    ) -> None:
        """Initialize the watershed segmentation base class.

        Args:
            img_grey: Input grayscale image to be segmented.
            img_rgb: Input RGB image for visualization.
            element_size: Size of structuring element for morphological operations.
            connectivity: Pixel connectivity for connected components (4 or 8).
            if_bknd_img: Flag indicating if background image should be used.
            bknd_img: Optional background image for background subtraction.
        """
        self.img_grey: npt.NDArray[np.int_] = img_grey
        self.img_grey_thresholded: npt.NDArray[np.bool_]
        self.img_grey_morph: npt.NDArray[np.int_]
        self.img_grey_dt: npt.NDArray[np.int_]
        self.img_rgb: npt.NDArray[np.int_] = img_rgb

        self.element_size: int = element_size
        self.connectivity: int = connectivity
        self.labels: npt.NDArray[np.int_]
        self.labels_watershed: npt.NDArray[np.int_]
        self.labels_on_img: npt.NDArray[np.int_]

        self.if_bknd_img: bool = if_bknd_img
        self.bknd_img: npt.NDArray[np.int_] = bknd_img

    def _threshold(self) -> None:
        """Apply thresholding to the input image.

        Uses either background subtraction thresholding if a background image is provided,
        or standard thresholding if no background image is available.
        """
        print("if_bknd in watershed_parent: ", self.if_bknd_img)
        threshold_methods = ThresholdMethods()
        if self.if_bknd_img is True:
            self.img_grey_thresholded = threshold_methods.threshold_with_background(self.img_grey, self.bknd_img)
        else:
            self.img_grey_thresholded = threshold_methods.threshold_without_background(self.img_grey)

    def _morph_process(self) -> None:
        """Apply morphological operations to the thresholded image.

        Uses the morphological_process function to clean up the binary image by
        filling holes and removing noise.
        """
        self.img_grey_morph = morphological_process(self.img_grey_thresholded, self.element_size)

    def _dist_transform(self) -> None:
        """Apply distance transform to the morphologically processed image.

        Computes the distance transform using L2 (Euclidean) distance metric.
        The result is converted to uint8 type for further processing.
        """
        self.img_grey_dt = cv2.distanceTransform(self.img_grey_morph, cv2.DIST_L2, self.element_size).astype(np.uint8)  # type: ignore

    def _initialize_labels(self, img: npt.NDArray[np.int_]) -> None:
        """Initialize label markers for watershed segmentation.

        Args:
            img: Binary image to be labeled using connected components analysis.
        """
        _, self.labels = cv2.connectedComponents(img, self.connectivity)  # type: ignore

    def _watershed_segmentation(self) -> None:
        """Apply watershed segmentation using the initialized labels.

        Uses OpenCV's watershed algorithm to segment the image based on the
        previously computed label markers.
        """
        self.labels_watershed = cv2.watershed(self.img_rgb, self.labels).astype(np.int_)

    def _overlay_labels_on_rgb(self) -> None:
        """Overlay the watershed segmentation labels on the RGB image.

        Creates a visualization of the segmentation results by overlaying
        the watershed labels on the original RGB image.
        """
        self.labels_on_img = overlay_labels_on_rgb(self.img_rgb, self.labels_watershed)
