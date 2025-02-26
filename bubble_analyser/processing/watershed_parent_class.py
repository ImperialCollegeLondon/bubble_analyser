import cv2
import numpy as np
from numpy import typing as npt
from typing import cast
from bubble_analyser.processing.image_postprocess import overlay_labels_on_rgb
from bubble_analyser.processing.morphological_process import morphological_process
from bubble_analyser.processing.threshold_methods import ThresholdMethods


class WatershedSegmentation:
    def __init__(
        self,
        img_grey: npt.NDArray[np.int_],
        img_rgb: npt.NDArray[np.int_],
        element_size: int = 5,
        connectivity: int = 4,
        if_bknd_img: bool = False,
        bknd_img: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], None),
    ) -> None:
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
        print("if_bknd in watershed_parent: ", self.if_bknd_img)
        threshold_methods = ThresholdMethods()
        if self.if_bknd_img is True:
            self.img_grey_thresholded = threshold_methods.threshold_with_background(
                self.img_grey, self.bknd_img
            )
        else:
            self.img_grey_thresholded = threshold_methods.threshold_without_background(
                self.img_grey
            )

    def _morph_process(self) -> None:
        self.img_grey_morph = morphological_process(
            self.img_grey_thresholded, self.element_size
        )

    def _dist_transform(self) -> None:
        self.img_grey_dt = cv2.distanceTransform(
            self.img_grey_morph, cv2.DIST_L2, self.element_size
        ).astype(np.uint8)

    def _initialize_labels(self, img: npt.NDArray[np.int_]) -> None:
        _, self.labels = cv2.connectedComponents(img, self.connectivity)  # type: ignore

    def _watershed_segmentation(self) -> None:
        self.labels_watershed = cv2.watershed(self.img_rgb, self.labels).astype(np.int_)

    def _overlay_labels_on_rgb(self) -> None:
        self.labels_on_img = overlay_labels_on_rgb(self.img_rgb, self.labels_watershed)
