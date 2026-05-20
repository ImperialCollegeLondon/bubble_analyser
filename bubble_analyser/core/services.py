"""Core services for Bubble Analyser."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt
from bubble_analyser.processing import Config, Image
from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskDetector


@dataclass
class AnalysisResult:
    """Result of an image analysis operation."""
    image_path: Path
    labels_on_img_before_filter: npt.NDArray[np.int_] | None = None
    ellipses_on_images: npt.NDArray[np.int_] | None = None
    img_rgb: npt.NDArray[np.int_] | None = None
    img_grey_morph_eroded: npt.NDArray[np.int_] | None = None
    labelled_ellipses_mask: npt.NDArray[np.int_] | None = None
    ellipses_properties: list[dict[str, Any]] | None = None
    bubble_count: int = 0
    success: bool = True
    error_message: str = ""


class AnalysisService:
    """Service class encapsulating the business logic for image analysis."""

    def __init__(self, config: Config):
        """Initialize the AnalysisService.

        Args:
            config: Configuration parameters for processing.
        """
        self.config = config
        self.algorithm = ""
        self.px2mm_display = 1.0
        self.bknd_img_path: Path | None = None
        
        self.detector: BubMaskDetector | None = None
        self.all_methods_n_params: dict[str, Any] = {}
        self.filter_param_dict_1: dict[str, Any] = {}
        self.filter_param_dict_2: dict[str, Any] = {}
        
    def setup_methods(self, methods_handler, filter_param_handler) -> None:
        """Setup the methods and filter parameters."""
        self.all_methods_n_params = methods_handler.full_dict
        self.filter_param_dict_1, self.filter_param_dict_2 = filter_param_handler.get_needed_params()

    def set_detector(self, detector: BubMaskDetector | None) -> None:
        """Set the deep learning detector."""
        self.detector = detector

    def process_image(self, image_path: Path, skip_filtering: bool = False) -> AnalysisResult:
        """Execute the full processing pipeline for a single image.

        Args:
            image_path: Path to the image to process.
            skip_filtering: If true, only run the segmentation step.

        Returns:
            AnalysisResult: Object containing the processed image outputs and properties.
        """
        result = AnalysisResult(image_path=image_path)
        
        try:
            # We import here to avoid circular imports if any, but better to inject MethodsHandler
            # For now, we will create the Image object
            from bubble_analyser.processing import MethodsHandler
            methods_handler = MethodsHandler(self.config)
            
            image = Image(
                self.px2mm_display,
                raw_img_path=image_path,
                all_methods_n_params=self.all_methods_n_params,
                methods_handler=methods_handler,
                bknd_img_path=self.bknd_img_path,
            )
            
            # Step 1: Segmentation
            image.processing_image_before_filtering(self.algorithm, self.detector)
            result.labels_on_img_before_filter = image.labels_on_img_before_filter
            
            # Step 2: Filtering (Optional)
            if not skip_filtering:
                image.load_filter_params(self.filter_param_dict_1, self.filter_param_dict_2)
                image.filtering_processing()
                image.get_ellipse_properties()
                
                result.ellipses_on_images = image.ellipses_on_images
                result.img_rgb = image.img_rgb
                result.img_grey_morph_eroded = image.img_grey_morph_eroded
                result.labelled_ellipses_mask = image.labelled_ellipses_mask
                result.ellipses_properties = image.ellipses_properties
                result.bubble_count = len(image.ellipses)

        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            result.success = False
            result.error_message = str(e)
            
        return result
