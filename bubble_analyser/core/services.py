"""Core services for Bubble Analyser."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy import typing as npt

from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskDetector
from bubble_analyser.core.models import ImageState
from bubble_analyser.processing import Config, MethodsHandler, image_preprocess

if TYPE_CHECKING:
    from bubble_analyser.processing import FilterParamHandler


@dataclass
class AnalysisResult:
    """Result of an image analysis operation."""

    image_path: Path
    labels_on_img_before_filter: npt.NDArray[Any] | None = None
    labels_before_filter: npt.NDArray[Any] | None = None
    ellipses_on_images: npt.NDArray[Any] | None = None
    img_rgb: npt.NDArray[Any] | None = None
    img_grey: npt.NDArray[Any] | None = None
    img_grey_morph_eroded: npt.NDArray[Any] | None = None
    labelled_ellipses_mask: npt.NDArray[Any] | None = None
    ellipses_properties: list[dict[str, Any]] | None = None
    bubble_count: int = 0
    success: bool = True
    error_message: str = ""


class PreprocessingService:
    """Service for handling image loading and resizing."""

    @staticmethod
    def load_and_resize(state: ImageState) -> ImageState:
        """Load and resize the raw and background images.

        Args:
            state: The current image state.

        Returns:
            ImageState: Updated image state with loaded arrays.
        """
        logging.info(f"Preprocessing image: {state.raw_img_path}")
        state.img_grey, state.img_rgb = image_preprocess(state.raw_img_path, state.resample)

        if state.bknd_img_path:
            state.bknd_img, _ = image_preprocess(state.bknd_img_path, state.resample)
            state.if_bknd_img = True

        return state


class SegmentationService:
    """Service for orchestrating image segmentation."""

    def __init__(self, methods_handler: MethodsHandler):
        self.methods_handler = methods_handler

    def segment(self, state: ImageState, algorithm: str, detector: BubMaskDetector | None = None) -> ImageState:
        """Run the selected segmentation algorithm on the image state.

        Args:
            state: Current image state.
            algorithm: Name of the algorithm to use.
            detector: Optional pre-initialized ML detector.

        Returns:
            ImageState: Updated state with labels.
        """
        from bubble_analyser.core.interfaces import SegmentationMethod

        all_params = state.all_methods_n_params if hasattr(state, "all_methods_n_params") else {}

        for name, instance in self.methods_handler.all_classes.items():
            if name == algorithm:
                params = all_params.get(name, {})
                method: SegmentationMethod = cast(SegmentationMethod, instance)

                method.initialize_processing(
                    params=params,
                    img_grey=state.img_grey,
                    img_rgb=state.img_rgb,
                    if_bknd_img=state.if_bknd_img,
                    px2mm=state.px2mm_display,
                    bknd_img=state.bknd_img,
                    cnn_model=detector,
                )

                res = method.get_results_img()
                state.labels_on_img_before_filter = res[0]
                state.labels_before_filter = res[1]
                state.img_grey_morph_eroded = res[2] if res[2] is not None else np.zeros((0, 0), dtype=np.uint8)

                break

        return state


class QuantificationService:
    """Service for filtering bubbles and extracting physical properties."""

    @staticmethod
    def process(
        state: ImageState, dict_params_1: dict[str, Any], dict_params_2: dict[str, Any], px2mm_display: float
    ) -> ImageState:
        """Run the quantification pipeline (filtering -> detection -> properties).

        Args:
            state: Current image state.
            dict_params_1: Filtering parameters.
            dict_params_2: Circle detection parameters.
            px2mm_display: Pixel to mm conversion factor.

        Returns:
            ImageState: Updated state with results.
        """
        from bubble_analyser.processing.circle_handler import EllipseHandler as CircleHandler

        labels_before_filter = state.labels_before_filter.copy()
        rgb_img = state.img_rgb.copy()

        handler = CircleHandler(labels_before_filter, rgb_img, px2mm_display, resample=state.resample)
        handler.load_filter_params(dict_params_1, dict_params_2)

        # 1. Filter
        state.labels_after_filter = handler.filter_labels_properties()

        # 2. Detect & Fill
        state.ellipses = handler.fill_ellipse_labels()

        # 3. Visualize
        state.ellipses_on_images = handler.overlay_ellipses_on_image()
        state.labelled_ellipses_mask = handler.create_labelled_image_from_ellipses()

        # 4. Properties
        state.ellipses_properties = handler.calculate_circle_properties()
        for ellipse in state.ellipses_properties:
            ellipse["filename"] = state.raw_img_path.name

        return state


class AnalysisService:
    """Service class encapsulating the business logic for image analysis."""

    def __init__(self, config: Config, methods_handler: MethodsHandler | None = None):
        """Initialize the AnalysisService.

        Args:
            config: Configuration parameters for processing.
            methods_handler: Optional pre-initialized methods handler.
        """
        self.config = config
        self.methods_handler = methods_handler if methods_handler else MethodsHandler(config)
        self.algorithm = ""
        self.px2mm_display = 1.0
        self.bknd_img_path: Path | None = None

        self.detector: BubMaskDetector | None = None
        self.all_methods_n_params: dict[str, Any] = {}
        self.filter_param_dict_1: dict[str, Any] = {}
        self.filter_param_dict_2: dict[str, Any] = {}

    def setup_methods(self, methods_handler: MethodsHandler, filter_param_handler: FilterParamHandler) -> None:
        """Setup the methods and filter parameters."""
        self.methods_handler = methods_handler
        self.all_methods_n_params = methods_handler.full_dict
        self.filter_param_dict_1, self.filter_param_dict_2 = filter_param_handler.get_needed_params()

    def set_detector(self, detector: BubMaskDetector | None) -> None:
        """Set the deep learning detector."""
        self.detector = detector

    def process_image(
        self,
        image_path: Path,
        skip_filtering: bool = False,
        existing_state: ImageState | None = None,
    ) -> AnalysisResult:
        """Execute the full processing pipeline for a single image.

        Args:
            image_path: Path to the image to process.
            skip_filtering: If true, only run the segmentation step.
            existing_state: If provided, skips preprocessing/segmentation and uses this state.

        Returns:
            AnalysisResult: Object containing the processed image outputs and properties.
        """
        result = AnalysisResult(image_path=image_path)

        try:
            if existing_state is None:
                # Create initial state
                state = ImageState(
                    raw_img_path=image_path,
                    px2mm_display=self.px2mm_display,
                    bknd_img_path=self.bknd_img_path,
                )

                # Step 1: Preprocessing
                params = self.all_methods_n_params.get(self.algorithm, {})
                state.resample = params.get("resample", 0.5)
                state = PreprocessingService.load_and_resize(state)

                # Step 2: Segmentation
                segmentation_service = SegmentationService(self.methods_handler)
                # Add all params to state temporarily for the service
                setattr(state, "all_methods_n_params", self.all_methods_n_params)
                state = segmentation_service.segment(state, self.algorithm, self.detector)

            else:
                state = existing_state

            result.labels_on_img_before_filter = state.labels_on_img_before_filter
            result.labels_before_filter = state.labels_before_filter
            result.img_rgb = state.img_rgb
            result.img_grey = state.img_grey
            result.img_grey_morph_eroded = state.img_grey_morph_eroded

            # Step 3: Quantification (Optional)
            if not skip_filtering:
                state = QuantificationService.process(
                    state,
                    self.filter_param_dict_1,
                    self.filter_param_dict_2,
                    self.px2mm_display,
                )

                result.ellipses_on_images = state.ellipses_on_images
                result.img_rgb = state.img_rgb
                result.img_grey_morph_eroded = state.img_grey_morph_eroded
                result.labelled_ellipses_mask = state.labelled_ellipses_mask
                result.ellipses_properties = state.ellipses_properties
                result.bubble_count = state.bubble_count

        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            result.success = False
            result.error_message = str(e)

        return result
