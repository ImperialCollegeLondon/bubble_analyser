"""Watershed Segmentation Methods for Bubble Analysis.

This module provides implementations of watershed segmentation algorithms for bubble analysis.
It includes two main approaches: Normal Watershed and Iterative Watershed. These methods
are designed to segment bubbles in images with different characteristics and requirements.

Classes:
    NormalWatershed: Implements the standard watershed algorithm with a single threshold.
    IterativeWatershed: Implements an advanced watershed algorithm that iteratively applies
                       thresholds to detect objects at different intensity levels.

Both classes inherit from the WatershedSegmentation parent class
"""

import logging
from typing import cast

import cv2
import numpy as np
from numpy import typing as npt

from bubble_analyser.processing.watershed_parent_class import WatershedSegmentation


class NormalWatershed(WatershedSegmentation):
    """Standard watershed segmentation implementation.

    This class implements the standard watershed algorithm for image segmentation.
    It uses a single threshold value to separate foreground and background regions,
    followed by watershed segmentation to identify individual objects.

    Attributes:
        name (str): Name identifier for this watershed method.
        img_grey_dt_thresh (npt.NDArray[np.int_]): Thresholded distance transform image.
        sure_fg (npt.NDArray[np.int_]): Sure foreground regions.
        sure_bg (npt.NDArray[np.int_]): Sure background regions.
        unknown (npt.NDArray[np.int_]): Unknown regions (neither sure foreground nor background).
        threshold_value (float): Threshold value for distance transform.
        resample (float): Resampling factor for image processing.
        if_bknd_img (bool): Flag indicating if background image is used.
    """

    def __init__(self, params: dict[str, float | int]) -> None:
        """Initialize the NormalWatershed segmentation method.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters for the watershed method.
                Must include 'resample', 'element_size', 'connectivity', and 'threshold_value'.
        """
        self.name = "Normal Watershed"
        self.img_grey_dt_thresh: npt.NDArray[np.int_]
        self.sure_fg: npt.NDArray[np.int_]
        self.sure_bg: npt.NDArray[np.int_]
        self.unknown: npt.NDArray[np.int_]
        self.threshold_value: float
        self.resample: float
        self.if_bknd_img: bool = False
        self.update_params(params)

    def get_needed_params(self) -> dict[str, float | int]:
        """Get the parameters required for this watershed method.

        Returns:
            dict[str, float | int]: Dictionary containing the required parameters and their current values.
        """
        return {
            "resample": self.resample,
            "element_size": self.element_size,
            "connectivity": self.connectivity,
            "threshold_value": self.threshold_value,
        }

    def initialize_processing(
        self,
        params: dict[str, float | int],
        img_grey: npt.NDArray[np.int_],
        img_rgb: npt.NDArray[np.int_],
        if_bknd_img: bool,
        bknd_img: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], None),
    ) -> None:
        """Initialize the processing with input images and parameters.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters for the watershed method.
            img_grey (npt.NDArray[np.int_]): Grayscale input image.
            img_rgb (npt.NDArray[np.int_]): RGB input image.
            if_bknd_img (bool): Flag indicating if background image is used.
            bknd_img (npt.NDArray[np.int_], optional): Background image if available. Defaults to None.
        """
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.if_bknd_img = if_bknd_img
        print("if_bknd_img in normal watershed: ", if_bknd_img)
        self.update_params(params)
        super().__init__(
            img_grey,
            img_rgb,
            if_bknd_img=if_bknd_img,
            bknd_img=bknd_img,
            element_size=self.element_size,
            connectivity=self.connectivity,
        )

    def update_params(self, params: dict[str, float | int]) -> None:
        """Update the parameters for the watershed method.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters to update.
                Must include 'resample', 'element_size', 'connectivity', and 'threshold_value'.
        """
        self.resample = params["resample"]
        self.element_size = params["element_size"]  # type: ignore
        self.connectivity = params["connectivity"]  # type: ignore
        self.threshold_value = params["threshold_value"]

    def __threshold_dt_image(self) -> None:
        """Apply threshold to the distance transform image.

        Uses the threshold_value parameter to create a binary image from the distance transform.
        The threshold is calculated as a fraction of the maximum value in the distance transform.
        """
        _, self.img_grey_dt_thresh = cv2.threshold(  # type: ignore
            self.img_grey_dt,
            self.threshold_value * self.img_grey_dt.max(),
            255,
            cv2.THRESH_BINARY,
        )

    def __get_sure_fg_bg(self) -> None:
        """Determine sure foreground and background regions.

        Creates masks for sure foreground (from thresholded distance transform),
        sure background (from dilated original image), and unknown regions (the difference
        between sure background and sure foreground).
        """
        sure_fg_initial = self.img_grey_dt_thresh.copy()

        self.sure_bg = np.array(
            cv2.dilate(self.img_grey, np.ones((3, 3), np.uint8), iterations=1),
            dtype=np.uint8,
        )  # type: ignore
        self.sure_fg = np.array(sure_fg_initial, dtype=np.uint8)  # type: ignore
        self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)  # type: ignore

    def get_results_img(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Execute the complete watershed segmentation process and return results.

        Performs the full sequence of operations for watershed segmentation:
        thresholding, morphological processing, distance transform, determining foreground/background,
        initializing labels, watershed segmentation, and overlaying results on the RGB image.

        Returns:
            tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: A tuple containing:
                - The RGB image with segmentation labels overlaid
                - The watershed segmentation labels array
        """
        self._threshold()
        self._morph_process()
        self._dist_transform()
        self.__threshold_dt_image()
        self.__get_sure_fg_bg()
        self._initialize_labels(self.sure_fg)
        self._watershed_segmentation()
        self._overlay_labels_on_rgb()

        return self.labels_on_img, self.labels_watershed


class IterativeWatershed(WatershedSegmentation):
    """Iterative watershed segmentation implementation.

    This class implements an advanced watershed algorithm that iteratively applies
    thresholds to detect objects at different intensity levels. It is particularly
    useful for images with objects of varying intensities or sizes.

    Attributes:
        name (str): Name identifier for this watershed method.
        max_thresh (float): Maximum threshold value for iterative process.
        min_thresh (float): Minimum threshold value for iterative process.
        step_size (float): Step size for decreasing threshold in each iteration.
        output_mask_for_labels (npt.NDArray[np.int_]): Final binary mask from iterative thresholding.
        no_overlap_count (int): Counter for non-overlapping objects detected.
        final_label_count (int): Total number of labels in the final segmentation.
    """

    def __init__(self, params: dict[str, float | int]) -> None:
        """Initialize the IterativeWatershed segmentation method.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters for the watershed method.
                Must include 'resample', 'element_size', 'connectivity', 'max_thresh', 'min_thresh',
                and 'step_size'.
        """
        self.name = "Iterative Watershed"
        self.max_thresh: float
        self.min_thresh: float
        self.step_size: float
        self.update_params(params)
        self.output_mask_for_labels: npt.NDArray[np.int_]
        self.no_overlap_count: int = 0  # Track number of "no overlap" occurrences
        self.final_label_count: int = 0  # Track final number of labels

    def get_needed_params(self) -> dict[str, float | int]:
        """Get the parameters required for this watershed method.

        Returns:
            dict[str, float | int]: Dictionary containing the required parameters and their current values.
        """
        return {
            "resample": self.resample,
            "element_size": self.element_size,
            "connectivity": self.connectivity,
            "max_thresh": self.max_thresh,
            "min_thresh": self.min_thresh,
            "step_size": self.step_size,
        }

    def initialize_processing(
        self,
        params: dict[str, float | int],
        img_grey: npt.NDArray[np.int_],
        img_rgb: npt.NDArray[np.int_],
        if_bknd_img: bool,
        bknd_img: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], None),
    ) -> None:
        """Initialize the processing with input images and parameters.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters for the watershed method.
            img_grey (npt.NDArray[np.int_]): Grayscale input image.
            img_rgb (npt.NDArray[np.int_]): RGB input image.
            if_bknd_img (bool): Flag indicating if background image is used.
            bknd_img (npt.NDArray[np.int_], optional): Background image if available. Defaults to None.
        """
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.if_bknd_img = if_bknd_img
        self.update_params(params)
        super().__init__(
            img_grey,
            img_rgb,
            if_bknd_img=if_bknd_img,
            bknd_img=bknd_img,
            element_size=self.element_size,
            connectivity=self.connectivity,
        )

    def update_params(self, params: dict[str, float | int]) -> None:
        """Update the parameters for the watershed method.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters to update.
                Must include 'resample', 'element_size', 'connectivity', 'max_thresh',
                'min_thresh', and 'step_size'.
        """
        self.resample = params["resample"]
        self.element_size = params["element_size"]  # type: ignore
        self.connectivity = params["connectivity"]  # type: ignore
        self.max_thresh = params["max_thresh"]
        self.min_thresh = params["min_thresh"]
        self.step_size = params["step_size"]

    def __iterative_threshold(self) -> None:
        """Apply iterative thresholding to detect objects at different intensity levels.

        This method iteratively applies decreasing thresholds to the distance transform image,
        detecting objects at each threshold level. It accumulates non-overlapping objects into
        a final mask, which is then used for watershed segmentation. This approach is effective
        for detecting objects with varying intensities or sizes that might be missed by a single
        threshold approach.

        The method starts at max_thresh and decreases by step_size until reaching min_thresh,
        keeping track of unique objects detected along the way.
        """
        logging.basicConfig(level=logging.INFO)

        image = self.img_grey_dt.astype(np.uint8)

        # Initialize the final mask to accumulate all detected objects
        output_mask = np.zeros_like(image, dtype=np.uint8)

        # Set the initial threshold
        current_thresh = self.max_thresh
        self.no_overlap_count = 0  # Reset counter

        while current_thresh >= self.min_thresh:
            # Apply binary thresholding
            _, thresholded = cv2.threshold(
                image, current_thresh * image.max(), 255, cv2.THRESH_BINARY
            )

            # Label the thresholded image
            num_labels, labels = cv2.connectedComponents(
                thresholded, connectivity=self.connectivity
            )
            logging.basicConfig(level=logging.DEBUG)
            logging.info(
                f"Threshold {current_thresh:.2f}: {num_labels} components found."
            )

            # Detect new objects by comparing with the final mask
            for label in range(1, num_labels):  # Skip label 0 (background)
                # Create a mask for the current label
                component_mask = (labels == label).astype(np.uint8) * 255

                # component_max_intensity = cv2.minMaxLoc(image, mask=component_mask)[1]
                # Check if the object is already in the final mask
                overlap = cv2.bitwise_and(output_mask, component_mask)

                if not np.any(overlap):  # If no overlap, it's a new object
                    self.no_overlap_count += 1
                    output_mask = cv2.bitwise_or(output_mask, component_mask * 255)  # type: ignore

            # Decrease the threshold for the next iteration
            current_thresh -= self.step_size
        self.output_mask_for_labels = output_mask  # type: ignore

        self.final_label_count, _ = cv2.connectedComponents(self.output_mask_for_labels)
        logging.info(
            f"Total unique labels in output_mask_for_labels: {self.final_label_count}"
        )
        logging.info(f"Total number of no overlap occurrences: {self.no_overlap_count}")

    def get_results_img(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Execute the complete iterative method process and return results.

        Performs the full sequence of operations for iterative watershed segmentation:
        thresholding, morphological processing, distance transform, iterative thresholding,
        initializing labels, watershed segmentation, and overlaying results on the RGB image.

        Returns:
            tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: A tuple containing:
                - The RGB image with segmentation labels overlaid
                - The watershed segmentation labels array
        """
        self._threshold()
        self._morph_process()
        self._dist_transform()
        self.__iterative_threshold()
        self._initialize_labels(self.output_mask_for_labels)
        self._watershed_segmentation()
        self._overlay_labels_on_rgb()
        return self.labels_on_img, self.labels_watershed


# if __name__ == "__main__":

#     # Define paths
#     img_grey_path = "../../tests/test_image_grey.JPG"
#     img_rgb_path = "../../tests/test_image_rgb.JPG"
#     output_path = "../../tests/test_image_segmented.JPG"
#       # Change to your desired output location
#     background_path = None  # Change if you have a background image

#     # Load images
#     img_rgb = cv2.imread(img_rgb_path)
#     if img_rgb is None:
#         raise ValueError(f"Error: Could not load image at {img_rgb_path}")

#     img_grey = cv2.imread(img_grey_path, cv2.IMREAD_GRAYSCALE)

#     # Load optional background image
#     bknd_img = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE) \
#                           if background_path else None

#     # Run Iterative Watershed Segmentation
#     iterative_watershed = IterativeWatershed(img_grey, img_rgb)

#     segmented_img, labels_watershed = iterative_watershed.run_segmentation()
#     pre_watershed_labels = iterative_watershed.output_mask_for_labels
#     np.save("../../tests/test_labels_watershed.npy", labels_watershed)
#     dist_transform = iterative_watershed.img_grey_dt

#     # Save and display results
#     plt.figure(figsize=(10, 5))
#     plt.subplot(331)
#     plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")

#     plt.subplot(332)
#     plt.imshow(segmented_img, cmap="jet")
#     plt.title("Segmented Image")

#     plt.subplot(333)
#     plt.imshow(pre_watershed_labels, cmap="gray")
#     plt.title("Pre Watershed Labels")

#     plt.subplot(334)
#     plt.imshow(labels_watershed, cmap="jet")
#     plt.title("Watershed Labels")

#     plt.subplot(335)
#     plt.imshow(dist_transform, cmap="gray")
#     plt.title("Distance Transform")
#     plt.savefig(output_path)
#     plt.show()

#     print(f"Segmentation completed! Output saved at: {output_path}")
