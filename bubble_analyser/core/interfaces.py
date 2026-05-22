"""Core interfaces for Bubble Analyser."""

from typing import Any, Protocol

import numpy as np
from numpy import typing as npt


class SegmentationMethod(Protocol):
    """Protocol defining the interface for all segmentation methods."""
    name: str
    description: str
    def initialize_processing(
        self,
        params: dict[str, Any],
        img_grey: npt.NDArray[np.uint8],
        img_rgb: npt.NDArray[np.uint8],
        if_bknd_img: bool,
        px2mm: float,
        bknd_img: npt.NDArray[np.uint8] | None = None,
        cnn_model: Any | None = None,
    ) -> None:
        """Initialize the processing with input images and parameters.

        Args:
            params: Dictionary containing parameters for the method.
            img_grey: Grayscale input image.
            img_rgb: RGB input image.
            if_bknd_img: Flag indicating if background image is used.
            px2mm: Conversion factor from pixels to millimeters.
            bknd_img: Background image if available.
            cnn_model: Optional pre-initialized CNN model for deep learning methods.
        """
        ...
    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], npt.NDArray[np.uint8]
    | None]:
        """Execute the segmentation process and return results.

        Returns:
            tuple containing:
                - labels_on_img: Image with overlaid labels for visualization.
                - labels_watershed: The raw integer label mask.
                - img_grey_morph_eroded: Morphological eroded image (or None).
        """
        ...
