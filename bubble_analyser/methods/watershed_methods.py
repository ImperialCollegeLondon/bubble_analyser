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
from cv2.typing import MatLike
from numpy import typing as npt

from bubble_analyser.processing.watershed_parent_class import WatershedSegmentation


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
        self.output_mask_for_labels: MatLike
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

    def _dilate_mask(self, mask: MatLike) -> MatLike:
        """Dilate the mask to enhance object boundaries.

        This method applies a morphological dilation operation to the input mask.
        The dilation operation enlarges the foreground regions, which helps in
        better defining the boundaries of objects.
        """
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)  # type: ignore
        return dilated_mask

    def __iterative_threshold(self, image: MatLike) -> tuple[MatLike, int]:
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

        image = image.astype(np.uint8)
        # Initialize the final mask to accumulate all detected objects
        output_mask = np.zeros_like(image, dtype=np.uint8)

        # Set the initial threshold
        current_thresh = self.max_thresh
        no_overlap_count = 0  # Reset counter

        while current_thresh >= self.min_thresh:
            # Apply binary thresholding
            _, thresholded = cv2.threshold(image, current_thresh * image.max(), 255, cv2.THRESH_BINARY)

            # Label the thresholded image
            num_labels, labels = cv2.connectedComponents(thresholded, connectivity=self.connectivity)
            logging.basicConfig(level=logging.DEBUG)
            logging.info(f"Threshold {current_thresh:.2f}: {num_labels} components found.")

            # Detect new objects by comparing with the final mask
            for label in range(1, num_labels):  # Skip label 0 (background)
                # Create a mask for the current label
                component_mask = (labels == label).astype(np.uint8) * 255

                # component_max_intensity = cv2.minMaxLoc(image, mask=component_mask)[1]
                # Check if the object is already in the final mask
                overlap = cv2.bitwise_and(output_mask, component_mask)

                if not np.any(overlap):  # If no overlap, it's a new object
                    no_overlap_count += 1
                    output_mask = cv2.bitwise_or(output_mask, component_mask * 255)  # type: ignore

            # Decrease the threshold for the next iteration
            current_thresh -= self.step_size

        final_label_count, _ = cv2.connectedComponents(output_mask)
        logging.info(f"Total unique labels in output_mask_for_labels: {final_label_count}")
        logging.info(f"Total number of no overlap occurrences: {no_overlap_count}")
        return output_mask, final_label_count

    def get_results_img(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Execute the complete iterative watershed segmentation process and return results.

        Performs the full sequence of operations:
        1. Initial thresholding of grayscale image
        2. Morphological processing to clean up thresholded image
        3. Distance transform calculation
        4. Iterative thresholding to detect objects at different intensity levels
        5. Mask dilation to enhance object boundaries
        6. Label initialization for watershed
        7. Watershed segmentation on morphologically processed image
        8. Filling detected objects with ellipses
        9. Overlay of final labels on original RGB image

        Returns:
            tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: A tuple containing:
                - The RGB image with segmentation labels overlaid
                - The filled watershed segmentation labels array
        """
        self.img_grey_thresholded = self._threshold(self.img_grey)
        self.img_grey_morph, self.img_grey_morph_eroded = self._morph_process(self.img_grey_thresholded)
        self.img_grey_dt = self._dist_transform(self.img_grey_morph)
        self.output_mask_for_labels, self.final_label_count = self.__iterative_threshold(self.img_grey_dt)
        self.dilated_mask = self._dilate_mask(self.output_mask_for_labels)
        self.labels = self._initialize_labels(self.dilated_mask)
        # img_grey_morph_rgb = cv2.cvtColor(self.img_grey_morph_eroded, cv2.COLOR_GRAY2RGB)  # type: ignore
        self.labels_watershed = self._watershed_segmentation(self.img_rgb, self.labels)
        self.labels_watershed_filled = self._fill_ellipses(self.labels_watershed)
        self.labels_on_img = self._overlay_labels_on_rgb(
            self.img_rgb, cast(npt.NDArray[np.int_], self.labels_watershed_filled)
        )
        return cast(npt.NDArray[np.int_], self.labels_on_img), cast(npt.NDArray[np.int_], self.labels_watershed_filled)


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
        self.name = "Default"
        self.img_grey_dt_thresh: MatLike
        self.sure_fg: npt.NDArray[np.uint8]
        self.sure_bg: npt.NDArray[np.uint8]
        self.unknown: MatLike
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
            "high_thresh": self.high_thresh,
            "mid_thresh": self.mid_thresh,
            "low_thresh": self.low_thresh,
            "element_size": self.element_size,
            "connectivity": self.connectivity,
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
                Must include 'resample', 'element_size', 'connectivity', and 'threshold_value'.
        """
        self.resample = params["resample"]
        self.high_thresh = params["high_thresh"]
        self.mid_thresh = params["mid_thresh"]
        self.low_thresh = params["low_thresh"]
        self.element_size = params["element_size"]  # type: ignore
        self.connectivity = params["connectivity"]  # type: ignore

    def __get_sure_fg_bg(
        self, target_image: npt.NDArray[np.int_], dt_thresh_image: MatLike
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], MatLike]:
        """Determine sure foreground and background regions.

        Creates masks for sure foreground (from thresholded distance transform),
        sure background (from dilated original image), and unknown regions (the difference
        between sure background and sure foreground).
        """
        sure_fg_initial = dt_thresh_image.copy()

        sure_bg = np.array(
            cv2.dilate(target_image, np.ones((3, 3), np.uint8), iterations=1),
            dtype=np.uint8,
        )  # type: ignore
        sure_fg = np.array(sure_fg_initial, dtype=np.uint8)  # type: ignore
        unknown = cv2.subtract(sure_bg, sure_fg)  # type: ignore

        return sure_fg, sure_bg, unknown

    def _three_way_threshold(
        self, target_image: MatLike, high_thresh: float, mid_thresh: float, low_thresh: float
    ) -> tuple[MatLike, int]:
        logging.basicConfig(level=logging.INFO)

        image = target_image.astype(np.uint8)
        # Initialize the final mask to accumulate all detected objects
        output_mask = np.zeros_like(image, dtype=np.uint8)

        # Set the initial threshold

        no_overlap_count = 0  # Reset counter
        thresh_list = [high_thresh, mid_thresh, low_thresh]
        current_thresh = high_thresh
        for i in range(3):
            current_thresh = thresh_list[i]

            # Apply binary thresholding
            _, thresholded = cv2.threshold(image, current_thresh * image.max(), 255, cv2.THRESH_BINARY)

            # Label the thresholded image
            num_labels, labels = cv2.connectedComponents(thresholded, connectivity=self.connectivity)
            logging.basicConfig(level=logging.DEBUG)
            logging.info(f"Threshold {current_thresh:.2f}: {num_labels} components found.")

            # Detect new objects by comparing with the final mask
            for label in range(1, num_labels):  # Skip label 0 (background)
                # Create a mask for the current label
                component_mask = (labels == label).astype(np.uint8) * 255

                # component_max_intensity = cv2.minMaxLoc(image, mask=component_mask)[1]
                # Check if the object is already in the final mask
                overlap = cv2.bitwise_and(output_mask, component_mask)

                if not np.any(overlap):  # If no overlap, it's a new object
                    no_overlap_count += 1
                    output_mask = cv2.bitwise_or(output_mask, component_mask * 255)  # type: ignore

            # Decrease the threshold for the next iteration

        final_label_count, _ = cv2.connectedComponents(output_mask)
        logging.info(f"Total unique labels in output_mask_for_labels: {final_label_count}")
        logging.info(f"Total number of no overlap occurrences: {no_overlap_count}")
        return output_mask, final_label_count

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
        self.img_grey_thresholded = self._threshold(self.img_grey)
        self.img_grey_morph, self.img_grey_morph_eroded = self._morph_process(self.img_grey_thresholded)
        self.img_grey_dt = self._dist_transform(self.img_grey_morph)
        self.img_grey_dt_thresh, self.final_label_count = self._three_way_threshold(
            self.img_grey_dt,
            self.high_thresh,
            self.mid_thresh,
            self.low_thresh,
        )
        self.sure_fg, self.sure_bg, self.unknown = self.__get_sure_fg_bg(self.img_grey, self.img_grey_dt_thresh)
        self.labels = self._initialize_labels(self.sure_fg)
        self.labels_watershed = self._watershed_segmentation(self.img_rgb, self.labels)
        # self.labels_watershed_filled = self._fill_ellipses(self.labels_watershed)
        self.labels_on_img = self._overlay_labels_on_rgb(
            self.img_rgb, cast(npt.NDArray[np.int_], self.labels_watershed)
        )

        return cast(npt.NDArray[np.int_], self.labels_on_img), cast(npt.NDArray[np.int_], self.labels_watershed)


# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     # Define paths
#     img_grey_path = "../../tests/test_image_grey.JPG"
#     img_rgb_path = "../../tests/test_image_rgb.JPG"
#     output_path = "../../tests/test_iterative_segmented_with_mt.JPG"
#     # Change to your desired output location
#     background_path = None  # Change if you have a background image

#     # Load images
#     img_rgb = cv2.imread(img_rgb_path)
#     if img_rgb is None:
#         raise ValueError(f"Error: Could not load image at {img_rgb_path}")

#     img_grey = cv2.imread(img_grey_path, cv2.IMREAD_GRAYSCALE)

#     # Load optional background image
#     bknd_img = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE) if background_path else None

#     params = {
#         "resample": 0.5,
#         "element_size": 5,
#         "connectivity": 8,
#         "max_thresh": 0.9,
#         "min_thresh": 0.05,
#         "step_size": 0.05,
#     }

#     # Run Iterative Watershed Segmentation without bknd img
#     iterative_watershed = IterativeWatershed(params)
#     iterative_watershed.initialize_processing(
#         params,
#         img_grey,  # type: ignore
#         img_rgb,  # type: ignore
#         if_bknd_img=False,
#     )

#     segmented_img, labels_watershed = iterative_watershed.get_results_img()
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

#     plt.subplot(334)
#     plt.imshow(labels_watershed, cmap="jet")
#     plt.title("Watershed Labels")

#     plt.subplot(335)
#     plt.imshow(dist_transform, cmap="gray")
#     plt.title("Distance Transform")

#     plt.subplot(333)
#     plt.imshow(iterative_watershed.output_mask_for_labels, cmap="gray")
#     plt.title("Output Mask")

#     plt.subplot(336)
#     plt.imshow(iterative_watershed.dilated_mask, cmap="gray")
#     plt.title("Dilated Mask")

#     plt.savefig(output_path)
#     plt.show()

#     # cv2.imwrite("../../tests/iterative_segmented_before_using_mt.JPG", iterative_watershed.labels_on_img)
#     cv2.imwrite("../../tests/iterative_segmented_using_mt.JPG", iterative_watershed.labels_on_img)
#     print(f"Segmentation completed! Output saved at: {output_path}")

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Define paths
    img_grey_path = "../../tests/test_image_grey.JPG"
    img_rgb_path = "../../tests/test_image_rgb.JPG"
    output_path = "../../tests/test_image_segmented_h20_t10_double_wts.JPG"
    # Change to your desired output location
    background_path = None  # Change if you have a background image

    # Load images
    img_rgb = cv2.imread(img_rgb_path)
    if img_rgb is None:
        raise ValueError(f"Error: Could not load image at {img_rgb_path}")

    img_grey = cv2.imread(img_grey_path, cv2.IMREAD_GRAYSCALE)

    # Load optional background image
    bknd_img = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE) if background_path else None

    params = {
        "resample": 0.5,
        "high_thresh": 0.9,
        "mid_thresh": 0.5,
        "low_thresh": 0.2,
        "h_value": 0.5,
        "element_size": 5,
        "connectivity": 4,
        "threshold_value": 0.1,
    }
    # Run Iterative Watershed Segmentation without bknd img
    normal_watershed = NormalWatershed(params)

    normal_watershed.initialize_processing(
        params,
        img_grey,  # type: ignore
        img_rgb,  # type: ignore
        if_bknd_img=False,
    )

    segmented_img, labels_watershed = normal_watershed.get_results_img()
    np.save("../../tests/test_labels_watershed.npy", labels_watershed)
    dist_transform = normal_watershed.img_grey_dt
    # dt_imhmin = normal_watershed.img_grey_dt_imhmin
    ch_labels = normal_watershed.labels_watershed

    # Save and display results
    plt.figure(figsize=(10, 5))
    plt.subplot(331)
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(332)
    plt.imshow(segmented_img, cmap="jet")
    plt.title("Segmented Image")

    plt.subplot(334)
    plt.imshow(dist_transform, cmap="gray")
    plt.title("Distance Transform")

    plt.subplot(335)
    plt.imshow(ch_labels, cmap="gray")
    plt.title("ch_labels")

    # plt.subplot(336)
    # plt.imshow(normal_watershed.b_mask, cmap="gray")
    # plt.title("s2_watershed")
    plt.savefig(output_path)
    plt.show()

    print(f"Segmentation completed! Output saved at: {output_path}")
