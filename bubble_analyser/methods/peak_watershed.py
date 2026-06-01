"""Peak-Based Watershed Segmentation Method for Bubble Analysis."""

from typing import Any, cast

import cv2
import numpy as np
from numpy import typing as npt
from skimage.feature import peak_local_max

from bubble_analyser.processing.watershed_parent_class import WatershedSegmentation


class PeakWatershed(WatershedSegmentation):
    """Segmentation method that uses local peak finding for bubble detection.

    Instead of global thresholds, this method identifies local maxima in the distance
    transform map to use as seeds for the watershed algorithm. This is more robust
    for images with varying bubble sizes and intensities.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize with default parameters."""
        self.name = "Peak-Based"
        self.description = (
            "Advanced method that finds bubble centers based on local height peaks "
            "in the distance transform. Best for varying bubble sizes."
        )
        self.min_distance_mm: float = 0.1
        self.sensitivity: float = 0.5
        self.smoothing_sigma: float = 1.0
        self.resample: float = 0.5
        self.px2mm: float = 1.0

        self.update_params(params)

    def get_needed_params(self) -> dict[str, Any]:
        """Return the parameters required by this method for the GUI."""
        return {
            "resample": self.resample,
            "min_distance_mm": self.min_distance_mm,
            "sensitivity": self.sensitivity,
            "smoothing_sigma": self.smoothing_sigma,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter for use in GUI tooltips.

        Returns:
            dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return {
            "resample": "Resampling factor to scale the image before processing. Lower values increase speed but reduce detail.",  # noqa: E501
            "min_distance_mm": "Minimum distance allowed between two separate bubble peaks.",
            "sensitivity": "Threshold sensitivity for identifying intensity peaks as potential bubbles.",
            "smoothing_sigma": "Standard deviation for Gaussian kernel used to smooth the distance transform map.",
        }

    def update_params(self, params: dict[str, Any]) -> None:
        """Update internal parameters from a dictionary."""
        self.resample = float(params.get("resample", self.resample))
        self.min_distance_mm = float(params.get("min_distance_mm", self.min_distance_mm))
        self.sensitivity = float(params.get("sensitivity", self.sensitivity))
        self.smoothing_sigma = float(params.get("smoothing_sigma", self.smoothing_sigma))

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
        """Set up images and sync parameters."""
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.if_bknd_img = if_bknd_img
        self.px2mm = px2mm
        self.update_params(params)

        # Initialize base class
        super().__init__(
            img_grey,
            img_rgb,
            element_size=3,  # Fixed small element for peaks
            connectivity=8,  # 8-connectivity is better for peak finding
            if_bknd_img=if_bknd_img,
            bknd_img=bknd_img,
        )

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
        """Run the peak-finding watershed pipeline."""
        # 1. Standard Thresholding
        self.img_grey_thresholded = self._threshold(self.img_grey)

        # 2. Morphological cleanup
        self.img_grey_morph, self.img_grey_morph_eroded = self._morph_process(self.img_grey_thresholded)

        # 3. Distance Transform (Float32)
        self.img_grey_dt = self._dist_transform(self.img_grey_morph)

        # 4. Optional Smoothing to prevent multiple peaks per bubble
        if self.smoothing_sigma > 0:
            dt_smoothed = cv2.GaussianBlur(self.img_grey_dt, (0, 0), self.smoothing_sigma)
        else:
            dt_smoothed = self.img_grey_dt

        # 5. Peak Finding
        # Convert mm to pixels: pixels = mm * (px/mm) * resample
        min_dist_px = max(1, int(self.min_distance_mm * self.px2mm * self.resample))

        # Sensitivity 1.0 means threshold at 100% of max, 0.0 means at 0%
        threshold_abs = (1.0 - self.sensitivity) * float(np.max(dt_smoothed))

        peaks = peak_local_max(
            dt_smoothed,
            min_distance=min_dist_px,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )

        # 6. Create markers for Watershed
        markers = np.zeros(dt_smoothed.shape, dtype=np.int32)
        # Background is Label 1
        markers[self.img_grey_morph == 0] = 1
        # Each peak is a unique seed starting from 2
        for i, (r, c) in enumerate(peaks):
            markers[r, c] = i + 2

        # 7. Watershed Segmentation
        self.labels_watershed = self._watershed_segmentation(self.img_rgb, markers)

        # 8. Post-process re-labeling
        self.labels_watershed_filled = self._fill_ellipses(self.labels_watershed)

        # 9. Visualization Overlay
        self.labels_on_img = self._overlay_labels_on_rgb(
            self.img_rgb, cast(npt.NDArray[np.int32], self.labels_watershed_filled)
        )

        return (
            cast(npt.NDArray[np.uint8], self.labels_on_img),
            cast(npt.NDArray[np.int32], self.labels_watershed_filled),
            cast(npt.NDArray[np.uint8], self.img_grey_morph_eroded),
        )
