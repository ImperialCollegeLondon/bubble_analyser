"""Example script demonstrating BubMask integration with Bubble Analyser.

This script shows how to use the BubMask wrapper and method within the Bubble Analyser
framework for bubble detection and analysis.
"""

import logging
import sys
from pathlib import Path

# Add bubble_analyser to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bubble_analyser.methods.bubmask_method import BubMaskWatershed
from bubble_analyser.methods.bubmask_wrapper import BubMaskConfig, BubMaskDetector, detect_bubbles_simple


def example_simple_detection():
    """Example of simple bubble detection using BubMask wrapper."""
    print("=== Simple BubMask Detection Example ===")

    # Paths (adjust these to your actual paths)
    weights_path = "c:/bubble_segment_cnn/BubMask/weights/mask_rcnn_bubble.h5"
    image_path = "c:/bubble_segment_cnn/BubMask/sample_image/IMG_6712.JPG"

    try:
        # Simple detection
        results = detect_bubbles_simple(image_path=image_path, weights_path=weights_path, confidence=0.9)

        print(f"Detected {results['bubble_count']} bubbles")
        print(f"Image shape: {results['image_shape']}")
        print(f"Confidence scores: {results['scores']}")

    except Exception as e:
        print(f"Error in simple detection: {e}")


def example_advanced_detection():
    """Example of advanced bubble detection using BubMask detector class."""
    print("\n=== Advanced BubMask Detection Example ===")

    # Paths (adjust these to your actual paths)
    weights_path = "c:/bubble_segment_cnn/BubMask/weights/mask_rcnn_bubble.h5"
    image_path = "c:/bubble_segment_cnn/BubMask/sample_image/IMG_6712.JPG"

    try:
        # Create custom configuration
        config = BubMaskConfig(confidence_threshold=0.85, image_min_dim=256, image_max_dim=512)

        # Initialize detector
        detector = BubMaskDetector(weights_path, config)

        # Detect bubbles
        results = detector.detect_bubbles(image_path, return_masks=True, return_splash=True)

        print(f"Detected {results['bubble_count']} bubbles")

        # Get bubble properties
        if "masks" in results:
            properties = detector.get_bubble_properties(results["masks"], pixel_to_mm=0.1)
            for i, props in enumerate(properties):
                print(f"Bubble {i + 1}:")
                print(f"  Area: {props['area_mm2']:.2f} mm²")
                print(f"  Diameter: {props['equivalent_diameter_mm']:.2f} mm")
                print(f"  Eccentricity: {props['eccentricity']:.3f}")

    except Exception as e:
        print(f"Error in advanced detection: {e}")


def example_batch_detection():
    """Example of batch bubble detection."""
    print("\n=== Batch BubMask Detection Example ===")

    # Paths (adjust these to your actual paths)
    weights_path = "c:/bubble_segment_cnn/BubMask/weights/mask_rcnn_bubble.h5"
    input_dir = "c:/bubble_segment_cnn/BubMask/sample_image"
    output_dir = "c:/bubble_segment_cnn/results/bubmask_batch"

    try:
        # Create detector
        config = BubMaskConfig(confidence_threshold=0.9)
        detector = BubMaskDetector(weights_path, config)

        # Batch detection
        results = detector.batch_detect(input_dir=input_dir, output_dir=output_dir, save_masks=True, save_splash=True)

        print(f"Processed {len(results)} images")
        for result in results:
            print(f"  {result['filename']}: {result['bubble_count']} bubbles")

    except Exception as e:
        print(f"Error in batch detection: {e}")


def example_watershed_method():
    """Example of using BubMask as a watershed method."""
    print("\n=== BubMask Watershed Method Example ===")

    # Paths (adjust these to your actual paths)
    weights_path = "c:/bubble_segment_cnn/BubMask/weights/mask_rcnn_bubble.h5"
    image_path = "c:/bubble_segment_cnn/BubMask/sample_image/IMG_6712.JPG"

    try:
        import cv2
        import skimage.io

        # Load image
        image = skimage.io.imread(image_path)
        if image.ndim == 2:
            image_rgb = cv2.merge((image, image, image))
            image_grey = image
        else:
            image_rgb = image
            image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Create watershed method
        params = {
            "weights_path": weights_path,
            "confidence_threshold": 0.9,
            "target_width": 800,
            "image_min_dim": 256,
            "image_max_dim": 512,
            "element_size": 3,
            "connectivity": 8,
        }

        watershed_method = BubMaskWatershed(params)

        # Initialize processing
        watershed_method.initialize_processing(params=params, img_grey=image_grey, img_rgb=image_rgb, if_bknd_img=False)

        # Run detection
        watershed_method.run_watershed()

        # Get results
        labels_on_img, labels_watershed = watershed_method.get_results_img()

        print(f"Watershed method detected {len(watershed_method.get_detection_confidence())} bubbles")
        print(f"Labels shape: {labels_watershed.shape}")
        print(f"Unique labels: {len(np.unique(labels_watershed)) - 1}")  # -1 for background

        # Get bubble properties
        properties = watershed_method.get_bubble_properties(pixel_to_mm=0.1)
        for i, props in enumerate(properties):
            print(f"Bubble {i + 1}: {props['area_mm2']:.2f} mm², diameter: {props['equivalent_diameter_mm']:.2f} mm")

    except Exception as e:
        print(f"Error in watershed method: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    print("BubMask Integration Examples")
    print("=" * 40)

    # Run examples
    example_simple_detection()
    example_advanced_detection()
    example_batch_detection()
    example_watershed_method()

    print("\n" + "=" * 40)
    print("Examples completed!")
