"""BubMask Integration Method for Bubble Analyser.

This module implements a watershed method that uses BubMask (Mask R-CNN) for bubble detection
instead of traditional watershed segmentation. It follows the same interface as other
watershed methods in the Bubble Analyser.

Classes:
    BubMaskWatershed: Watershed method using BubMask for detection.
"""

import logging
import os
from typing import cast, Dict, Any
import numpy as np
from numpy import typing as npt
import cv2
import time
from pathlib import Path

from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskDetector, BubMaskConfig


class BubMaskWatershed():

    def __init__(self, params: Dict[str, float | int]) -> None:
        """Initialize the BubMask watershed segmentation method.

        Args:
            params (Dict[str, float | int]): Dictionary containing parameters for the method.
                Must include 'weights_path', 'confidence_threshold', 'target_width',
                'image_min_dim', 'image_max_dim', 'element_size', and 'connectivity'.
        """
        self.name = "BubMask (Deep Learning)"
        self.description = "Detection of bubbles using Mask R-CNN model from MULTIPHASE FLOW & FLOW VISUALIZATION LAB, https://doi.org/10.1038/s41598-021-88334-0 (Kim & Park, 2021)."
        base_dir = str(Path(__file__).resolve().parents[2])
        self.weights_path: str = os.path.join(base_dir, "bubble_analyser/weights/mask_rcnn_bubble.h5")
        self.confidence_threshold: float = 0.9
        self.resample: float = 0.5
        self.target_width: int = 1000
        self.image_min_dim: int = 192
        self.image_max_dim: int = 384

        self.detector: BubMaskDetector | None = None
        self.detection_results: Dict = {}
        
        # Initialize parent class attributes
        super().__init__()
        logging.info("Initializing BubMask detector...")
        # Update with provided parameters
        self.update_params(params)

    def get_needed_params(self) -> Dict[str, float | int | str]:
        """Get the parameters required for this watershed method.

        Returns:
            Dict[str, float | int | str]: Dictionary containing the required parameters and their current values.
        """
        return {
            "weights_path": self.weights_path,
            "confidence_threshold": self.confidence_threshold,
            # "target_width": self.target_width,
            "resample": self.resample,
            "image_min_dim": self.image_min_dim,
            "image_max_dim": self.image_max_dim,
        }

    def update_params(self, params: Dict[str, Any]) -> None:
        """Update the parameters for this watershed method.

        Args:
            params (Dict[str, Any]): Dictionary containing parameters to update.
        """
        # Update BubMask-specific parameters
        if "weights_path" in params:
            self.weights_path = str(params["weights_path"])
        if "confidence_threshold" in params:
            self.confidence_threshold = float(params["confidence_threshold"])
        # if "target_width" in params:
        #     self.target_width = int(params["target_width"])
        if "resample" in params:
            self.resample = float(params["resample"])
        if "image_min_dim" in params:
            self.image_min_dim = int(params["image_min_dim"])
        if "image_max_dim" in params:
            self.image_max_dim = int(params["image_max_dim"])

    def initialize_processing(
        self,
        params: Dict[str, float | int],
        img_grey: npt.NDArray[np.int_],
        img_rgb: npt.NDArray[np.int_],
        if_bknd_img: bool,
        bknd_img: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], None),
        cnn_model: Any = None,
    ) -> None:
        """Initialize the processing with input images and parameters.

        Args:
            params (Dict[str, float | int]): Dictionary containing parameters for the method.
            img_grey (npt.NDArray[np.int_]): Grayscale input image.
            img_rgb (npt.NDArray[np.int_]): RGB input image.
            if_bknd_img (bool): Flag indicating if background image is used.
            bknd_img (npt.NDArray[np.int_], optional): Background image if available.
        """
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.if_bknd_img = if_bknd_img
        self.bknd_img = bknd_img
        
        # Update parameters
        self.update_params(params)
        self.detector = cnn_model
        # Initialize BubMask detector if not already done
        if self.detector is None and self.weights_path:
            try:
                config = BubMaskConfig(
                    confidence_threshold=self.confidence_threshold,
                    image_min_dim=self.image_min_dim,
                    image_max_dim=self.image_max_dim
                )
                self.detector = BubMaskDetector(self.weights_path, config)
                logging.info("BubMask detector initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize BubMask detector: {e}")
                raise

    def detect_bubbles(self) -> None:
        """Run the BubMask detection process.
        
        This method replaces traditional watershed segmentation with BubMask detection.
        """
        if self.detector is None:
            raise RuntimeError("BubMask detector not initialized. Call initialize_processing first.")
        
        try:
            # Save the RGB image temporarily for BubMask processing
            import tempfile
            import skimage.io
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                skimage.io.imsave(temp_path, self.img_rgb)
            
            
            # Run BubMask detection
            self.detection_results = self.detector.detect_bubbles(
                temp_path, 
                return_masks=True, 
                return_splash=False
            )
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            # Convert BubMask results to watershed format
            self._convert_bubmask_to_watershed()
            
            logging.info(f"BubMask detected {self.detection_results['bubble_count']} bubbles")
            
        except Exception as e:
            logging.error(f"Error running BubMask detection: {e}")
            raise

    def _convert_bubmask_to_watershed(self) -> None:
        """Convert BubMask detection results to watershed segmentation format."""
        if not self.detection_results or 'masks' not in self.detection_results:
            # No bubbles detected, create empty labels
            self.labels_watershed = np.zeros(self.img_grey.shape, dtype=np.int32)
            return
        
        masks = self.detection_results['masks']
        height, width = self.img_grey.shape
        
        # Create labels array
        self.labels_watershed = np.zeros((height, width), dtype=np.int32)
        
        # Convert each mask to a labeled region
        for i in range(masks.shape[2]):
            mask = masks[:, :, i]
            # Resize mask to match original image size if needed
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Add to labels (i+1 because 0 is background)
            self.labels_watershed[mask > 0] = i + 1

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32]]:
        """Get the results image with overlaid bubble detection.

        Returns:
            tuple: (labels_on_img, labels_watershed) where labels_on_img is the 
                  overlayed image with bubble masks and labels_watershed is just the masks.
        """
        start_time = time.time()
        logging.info("Running BubMask detection...")
        self.detect_bubbles()
        end_time = time.time()
        logging.info(f"BubMask detection completed in {end_time - start_time:.2f} seconds")


        # Check if BubMask detection results are available
        if not hasattr(self, 'labels_watershed'):
            raise RuntimeError("BubMask detection not run yet. Call detect_bubbles first.")
        
        # Create overlay image
        labels_on_img = self.img_rgb.copy()
        
        # Generate colors for each bubble
        num_bubbles = np.max(self.labels_watershed)
        if num_bubbles > 0:
            colors = np.random.randint(0, 255, size=(num_bubbles + 1, 3))
            colors[0] = [0, 0, 0]  # Background is black
            
            # Create colored overlay
            for label_id in range(1, num_bubbles + 1):
                mask = self.labels_watershed == label_id
                labels_on_img[mask] = colors[label_id]
        
        return labels_on_img.astype(np.uint8), self.labels_watershed, None

    def get_bubble_properties(self, pixel_to_mm: float = 1.0) -> list[Dict]:
        """Get detailed properties of detected bubbles.

        Args:
            pixel_to_mm (float): Conversion factor from pixels to millimeters.

        Returns:
            list[Dict]: List of dictionaries containing bubble properties.
        """
        if not self.detection_results or 'masks' not in self.detection_results:
            return []
        
        return self.detector.get_bubble_properties(
            self.detection_results['masks'], 
            pixel_to_mm
        )

    def get_detection_confidence(self) -> list[float]:
        """Get confidence scores for detected bubbles.

        Returns:
            list[float]: List of confidence scores for each detected bubble.
        """
        if not self.detection_results or 'scores' not in self.detection_results:
            return []
        
        return self.detection_results['scores'].tolist()

    def get_bounding_boxes(self) -> list[tuple]:
        """Get bounding boxes for detected bubbles.

        Returns:
            list[tuple]: List of bounding boxes as (y1, x1, y2, x2) tuples.
        """
        if not self.detection_results or 'rois' not in self.detection_results:
            return []
        
        return [tuple(roi) for roi in self.detection_results['rois']]