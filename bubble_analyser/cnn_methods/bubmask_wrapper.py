"""BubMask Integration Wrapper for Bubble Analyser.

This module provides a clean interface to integrate BubMask (Mask R-CNN) functionality
into the Bubble Analyser application. It wraps the BubMask detection and processing
functions to make them easily callable from other programs.

Classes:
    BubMaskDetector: Main wrapper class for BubMask functionality.
    BubMaskConfig: Configuration class for BubMask parameters.

Usage:
    from bubble_analyser.methods.bubmask_wrapper import BubMaskDetector
    
    detector = BubMaskDetector(weights_path="path/to/weights.h5")
    results = detector.detect_bubbles(image_path="path/to/image.jpg")
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
import skimage.io
from numpy import typing as npt

# Add BubMask to Python path
CURRENT_DIR = Path(__file__).parent.absolute()
BUBMASK_DIR = CURRENT_DIR.parent.parent.parent / "BubMask"
BUBBLE_DIR = BUBMASK_DIR / "bubble"

if str(BUBMASK_DIR) not in sys.path:
    sys.path.insert(0, str(BUBMASK_DIR))
if str(BUBBLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUBBLE_DIR))

try:
    # Import BubMask modules
    from bubble_analyser.mrcnn import model as modellib
    from bubble_analyser.bubble import BubbleConfig, _InfConfig, color_splash
    import bubble
except ImportError as e:
    logging.error(f"Failed to import BubMask modules: {e}")
    raise ImportError(f"BubMask modules not found. Please ensure BubMask is properly installed. Error: {e}")


class BubMaskConfig:
    """Configuration class for BubMask parameters."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.9,
                 image_min_dim: int = 128,
                 image_max_dim: int = 256,
                 gpu_count: int = 1,
                 images_per_gpu: int = 1):
        """Initialize BubMask configuration.
        
        Args:
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            image_min_dim: Minimum image dimension for processing
            image_max_dim: Maximum image dimension for processing
            gpu_count: Number of GPUs to use (1 for single GPU, 0 for CPU)
            images_per_gpu: Number of images per GPU batch
        """
        self.confidence_threshold = confidence_threshold
        self.image_min_dim = image_min_dim
        self.image_max_dim = image_max_dim
        self.gpu_count = gpu_count
        self.images_per_gpu = images_per_gpu
    
    @classmethod
    def for_high_quality(cls, confidence_threshold: float = 0.9):
        """Configuration for high quality processing (requires more GPU memory).
        
        Args:
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            BubMaskConfig: High quality configuration
        """
        return cls(
            confidence_threshold=confidence_threshold,
            image_min_dim=512,
            image_max_dim=1024,
            gpu_count=1,
            images_per_gpu=1
        )
    
    @classmethod
    def for_medium_quality(cls, confidence_threshold: float = 0.9):
        """Configuration for medium quality processing (balanced memory usage).
        
        Args:
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            BubMaskConfig: Medium quality configuration
        """
        return cls(
            confidence_threshold=confidence_threshold,
            image_min_dim=256,
            image_max_dim=512,
            gpu_count=1,
            images_per_gpu=1
        )
    
    @classmethod
    def for_low_memory_gpu(cls, confidence_threshold: float = 0.9):
        """Configuration for low memory GPU processing.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            BubMaskConfig: Low memory GPU configuration
        """
        return cls(
            confidence_threshold=confidence_threshold,
            image_min_dim=128,
            image_max_dim=256,
            gpu_count=1,
            images_per_gpu=1
        )
    
    @classmethod
    def for_cpu_only(cls, confidence_threshold: float = 0.9):
        """Configuration for CPU-only processing (slower but no memory limits).
        
        Args:
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            BubMaskConfig: CPU-only configuration
        """
        return cls(
            confidence_threshold=confidence_threshold,
            image_min_dim=256,
            image_max_dim=512,
            gpu_count=0,  # Force CPU usage
            images_per_gpu=1
        )
    
    @classmethod
    def get_progressive_configs(cls, confidence_threshold: float = 0.9):
        """Get a list of configurations to try progressively from highest to lowest quality.
        
        This allows automatic fallback when GPU memory is insufficient.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List[BubMaskConfig]: Configurations ordered from highest to lowest quality
        """
        return [
            cls.for_high_quality(confidence_threshold),    # 512x1024 - Best quality
            cls.for_medium_quality(confidence_threshold),  # 256x512  - Good balance
            cls.for_low_memory_gpu(confidence_threshold),  # 128x256  - Memory safe
        ]


class BubMaskDetector:
    """Main wrapper class for BubMask bubble detection functionality."""
    
    def __init__(self, weights_path: str, config: Optional[BubMaskConfig] = None):
        """Initialize the BubMask detector.
        
        Args:
            weights_path: Path to the trained model weights (.h5 file)
            config: BubMask configuration object
        """
        self.weights_path = Path(weights_path)
        self.config = config or BubMaskConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the BubMask model with specified configuration."""
        try:
            # Create inference configuration
            inference_config = _InfConfig()
            inference_config.IMAGE_MIN_DIM = self.config.image_min_dim
            inference_config.IMAGE_MAX_DIM = self.config.image_max_dim
            inference_config.GPU_COUNT = self.config.gpu_count
            inference_config.IMAGES_PER_GPU = self.config.images_per_gpu
            
            # Create model object in inference mode
            self.model = modellib.MaskRCNN(mode="inference", 
                                         model_dir="logs", 
                                         config=inference_config)
            
            # Load weights
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
            
            self.model.load_weights(str(self.weights_path), by_name=True)
            logging.info(f"BubMask model loaded successfully from {self.weights_path}")
            
        except Exception as e:
            logging.error(f"Failed to load BubMask model: {e}")
            raise
    
    def detect_bubbles_progressive(self, 
                                 image_path: Union[str, Path], 
                                 return_masks: bool = True,
                                 return_splash: bool = False,
                                 confidence_threshold: float = 0.9) -> Dict:
        """Detect bubbles using progressive quality settings.
        
        Automatically tries higher quality settings first and falls back to lower
        quality if GPU memory errors occur.
        
        Args:
            image_path: Path to the input image
            return_masks: Whether to return detection masks
            return_splash: Whether to return color splash image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dict: Detection results with additional 'config_used' field
        """
        configs = BubMaskConfig.get_progressive_configs(confidence_threshold)
        
        for i, config in enumerate(configs):
            try:
                # Create a new detector with this configuration
                temp_detector = BubMaskDetector(self.weights_path, config)
                
                # Try detection
                results = temp_detector.detect_bubbles(
                    image_path, 
                    return_masks=return_masks, 
                    return_splash=return_splash
                )
                
                # Add information about which config was used
                quality_names = ["high_quality", "medium_quality", "low_memory"]
                results['config_used'] = quality_names[i]
                results['image_dimensions'] = f"{config.image_min_dim}x{config.image_max_dim}"
                
                if i > 0:
                    logging.warning(f"Used {quality_names[i]} configuration due to memory constraints")
                else:
                    logging.info(f"Successfully used {quality_names[i]} configuration")
                
                return results
                
            except Exception as e:
                error_msg = str(e).lower()
                if "oom" in error_msg or "memory" in error_msg or "resource_exhausted" in error_msg:
                    if i < len(configs) - 1:
                        logging.warning(f"GPU memory error with {config.image_min_dim}x{config.image_max_dim}, trying lower quality...")
                        continue
                    else:
                        logging.error("All quality configurations failed due to memory constraints")
                        raise
                else:
                    # Non-memory error, don't retry
                    raise
        
        raise RuntimeError("All progressive quality configurations failed")
    
    def detect_bubbles(self, 
                      image_path: Union[str, Path], 
                      return_masks: bool = True,
                      return_splash: bool = False) -> Dict:
        """Detect bubbles in a single image.
        
        Args:
            image_path: Path to the input image
            return_masks: Whether to return individual bubble masks
            return_splash: Whether to return color splash image
            
        Returns:
            Dictionary containing detection results:
            - 'rois': Bounding boxes [N, (y1, x1, y2, x2)]
            - 'class_ids': Class IDs [N]
            - 'scores': Confidence scores [N]
            - 'masks': Binary masks [H, W, N] (if return_masks=True)
            - 'splash': Color splash image [H, W, 3] (if return_splash=True)
            - 'bubble_count': Number of detected bubbles
            - 'image_shape': Original image shape
        """
        try:
            # Load and preprocess image
            image = skimage.io.imread(str(image_path))
            if image.ndim == 2:
                image = cv2.merge((image, image, image))
            
            # Run detection
            results = self.model.detect([image], verbose=0)[0]
            
            # Filter by confidence threshold
            keep_indices = results['scores'] >= self.config.confidence_threshold
            filtered_results = {
                'rois': results['rois'][keep_indices],
                'class_ids': results['class_ids'][keep_indices],
                'scores': results['scores'][keep_indices],
                'bubble_count': np.sum(keep_indices),
                'image_shape': image.shape
            }
            
            if return_masks:
                filtered_results['masks'] = results['masks'][:, :, keep_indices]
            
            if return_splash:
                splash = color_splash(image, results['masks'][:, :, keep_indices])
                filtered_results['splash'] = splash
            
            logging.info(f"Detected {filtered_results['bubble_count']} bubbles in {image_path}")
            return filtered_results
            
        except Exception as e:
            logging.error(f"Error detecting bubbles in {image_path}: {e}")
            raise

    def batch_detect_parallel(self, 
                            input_dir: Union[str, Path], 
                            output_dir: Optional[Union[str, Path]] = None,
                            save_masks: bool = True,
                            save_splash: bool = True,
                            batch_size: int = 4,
                            use_multiprocessing: bool = False,
                            max_workers: int = 4) -> List[Dict]:
        """Detect bubbles in multiple images using parallel processing.
        
        On GPU (default), uses batch processing to maximize throughput.
        On CPU (use_multiprocessing=True), uses process pool for parallelism.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results (optional)
            save_masks: Whether to save individual masks
            save_splash: Whether to save splash images
            batch_size: Number of images to process at once (GPU mode)
            use_multiprocessing: Whether to use CPU multiprocessing (CPU mode only)
            max_workers: Number of worker processes (CPU mode only)
            
        Returns:
            List of detection results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logging.warning(f"No image files found in {input_path}")
            return []
        
        logging.info(f"Found {len(image_files)} images to process")
        results = []
        
        if use_multiprocessing and self.config.gpu_count == 0:
            # CPU Multiprocessing Mode
            logging.info(f"Starting CPU multiprocessing with {max_workers} workers")
            logging.warning("Multiprocessing requires complex TF setup. "
                          "Falling back to batch processing which is efficient for both CPU and GPU.")
            
        # Batch Processing Mode (Works for both GPU and CPU)
        logging.info(f"Starting batch processing with batch_size={batch_size}")
        
        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            valid_files = []
            
            # Load images
            for img_file in batch_files:
                try:
                    image = skimage.io.imread(str(img_file))
                    if image.ndim == 2:
                        image = cv2.merge((image, image, image))
                    # Handle alpha channel if present
                    if image.shape[-1] == 4:
                        image = image[..., :3]
                    batch_images.append(image)
                    valid_files.append(img_file)
                except Exception as e:
                    logging.error(f"Failed to load {img_file}: {e}")
            
            if not batch_images:
                continue
                
            try:
                # Run detection on batch
                batch_results = self.model.detect(batch_images, verbose=0)
                
                for j, result in enumerate(batch_results):
                    img_file = valid_files[j]
                    
                    # Filter by confidence
                    keep_indices = result['scores'] >= self.config.confidence_threshold
                    filtered_result = {
                        'rois': result['rois'][keep_indices],
                        'class_ids': result['class_ids'][keep_indices],
                        'scores': result['scores'][keep_indices],
                        'bubble_count': np.sum(keep_indices),
                        'image_shape': batch_images[j].shape,
                        'filename': img_file.name
                    }
                    
                    if save_masks:
                        filtered_result['masks'] = result['masks'][:, :, keep_indices]
                    
                    if save_splash:
                        splash = color_splash(batch_images[j], result['masks'][:, :, keep_indices])
                        filtered_result['splash'] = splash
                    
                    results.append(filtered_result)
                    
                    # Save results
                    if output_dir:
                        self._save_results(filtered_result, img_file, output_dir, save_masks, save_splash)
                        
                logging.info(f"Processed batch {i//batch_size + 1}/{(len(image_files)+batch_size-1)//batch_size}")
                
            except Exception as e:
                logging.error(f"Error processing batch starting at {batch_files[0]}: {e}")
                # Fallback to single image processing for this batch
                for img_file in batch_files:
                    try:
                        res = self.detect_bubbles(img_file, return_masks=save_masks, return_splash=save_splash)
                        res['filename'] = img_file.name
                        results.append(res)
                        if output_dir:
                            self._save_results(res, img_file, output_dir, save_masks, save_splash)
                    except Exception as inner_e:
                        logging.error(f"Fallback failed for {img_file}: {inner_e}")

        return results

    def batch_detect(self, 
                    input_dir: Union[str, Path], 
                    output_dir: Optional[Union[str, Path]] = None,
                    save_masks: bool = True,
                    save_splash: bool = True) -> List[Dict]:
        """Detect bubbles in multiple images from a directory.
        
        This method now uses batch processing by default for better performance.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results (optional)
            save_masks: Whether to save individual masks
            save_splash: Whether to save splash images
            
        Returns:
            List of detection results for each image
        """
        return self.batch_detect_parallel(
            input_dir=input_dir,
            output_dir=output_dir,
            save_masks=save_masks,
            save_splash=save_splash,
            batch_size=4  # Default batch size
        )

    def _save_results(self, 
                     result: Dict, 
                     image_file: Path, 
                     output_dir: Union[str, Path],
                     save_masks: bool,
                     save_splash: bool) -> None:
        """Save detection results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = image_file.stem
        
        # Save splash image
        if save_splash and 'splash' in result:
            splash_path = output_path / f"splash_{base_name}.png"
            skimage.io.imsave(str(splash_path), result['splash'])
        
        # Save individual masks
        if save_masks and 'masks' in result:
            masks = result['masks']
            for i in range(masks.shape[2]):
                mask_path = output_path / f"mask_{base_name}_bubble_{i:03d}.png"
                mask = (masks[:, :, i] * 255).astype(np.uint8)
                skimage.io.imsave(str(mask_path), mask)

    def get_bubble_properties(self, masks: npt.NDArray, pixel_to_mm: float = 1.0) -> List[Dict]:
        """Calculate properties of detected bubbles.
        
        Args:
            masks: Binary masks array [H, W, N]
            pixel_to_mm: Conversion factor from pixels to millimeters
            
        Returns:
            List of dictionaries containing bubble properties
        """
        from skimage.measure import regionprops, label
        
        properties = []
        for i in range(masks.shape[2]):
            mask = masks[:, :, i].astype(np.uint8)
            labeled_mask = label(mask)
            props = regionprops(labeled_mask)
            
            if props:
                prop = props[0]  # Should only be one region per mask
                bubble_props = {
                    'area_pixels': prop.area,
                    'area_mm2': prop.area * (pixel_to_mm ** 2),
                    'centroid': prop.centroid,
                    'equivalent_diameter_pixels': prop.equivalent_diameter,
                    'equivalent_diameter_mm': prop.equivalent_diameter * pixel_to_mm,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'orientation': prop.orientation,
                    'major_axis_length': prop.major_axis_length * pixel_to_mm,
                    'minor_axis_length': prop.minor_axis_length * pixel_to_mm,
                }
                properties.append(bubble_props)
        
        return properties
    
    def update_config(self, **kwargs) -> None:
        """Update detector configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logging.warning(f"Unknown configuration parameter: {key}")


# Convenience functions for easy integration
def detect_bubbles_simple(image_path: str, 
                         weights_path: str, 
                         confidence: float = 0.9) -> Dict:
    """Simple function to detect bubbles in an image.
    
    Args:
        image_path: Path to input image
        weights_path: Path to model weights
        confidence: Confidence threshold
        
    Returns:
        Detection results dictionary
    """
    config = BubMaskConfig(confidence_threshold=confidence)
    detector = BubMaskDetector(weights_path, config)
    return detector.detect_bubbles(image_path, return_masks=True, return_splash=True)


def batch_detect_simple(input_dir: str, 
                       weights_path: str, 
                       output_dir: str,
                       confidence: float = 0.9) -> List[Dict]:
    """Simple function to detect bubbles in multiple images.
    
    Args:
        input_dir: Directory containing input images
        weights_path: Path to model weights
        output_dir: Directory to save results
        confidence: Confidence threshold
        
    Returns:
        List of detection results
    """
    config = BubMaskConfig(confidence_threshold=confidence)
    detector = BubMaskDetector(weights_path, config)
    return detector.batch_detect(input_dir, output_dir)