#!/usr/bin/env python3
"""
Test script for BubMask CNN bubble detection with progressive quality settings.
Now includes conversion to Bubble Analyser compatible labeled masks.
"""

import warnings
# Suppress low contrast image warningwarnings from skimage
warnings.filterwarnings("ignore", message=".*is a low contrast image.*")

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from skimage import io

# Add the parent directory to the path to import bubmask_wrapper
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from bubmask_wrapper import BubMaskDetector, BubMaskConfig

def convert_bubmask_to_watershed_labels(masks, target_shape):
    """Convert BubMask 3D masks to Bubble Analyser compatible 2D labeled masks.
    
    This function replicates the conversion process from bubmask_method.py
    to transform BubMask's 3D mask format into watershed-style 2D labels.
    
    Args:
        masks (np.ndarray): 3D array of shape (height, width, num_detections) 
                           containing individual bubble masks
        target_shape (tuple): Target shape (height, width) for the output labels
        
    Returns:
        np.ndarray: 2D labeled array where each bubble has a unique integer ID
                   (0 = background, 1,2,3... = individual bubbles)
    """
    if masks is None or masks.size == 0:
        # No bubbles detected, create empty labels
        return np.zeros(target_shape, dtype=np.int32)
    
    height, width = target_shape
    
    # Create labels array - single 2D array compatible with watershed format
    labels_watershed = np.zeros((height, width), dtype=np.int32)
    
    # Convert each mask to a labeled region
    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        
        # Resize mask to match target image size if needed
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Add to labels (i+1 because 0 is background)
        # Note: Later detections will overwrite earlier ones in overlapping regions
        labels_watershed[mask > 0] = i + 1
    
    return labels_watershed

def save_watershed_labels(labels, output_path, image_name):
    """Save the watershed-style labels as both image and numpy array.
    
    Args:
        labels (np.ndarray): 2D labeled array
        output_path (str): Output directory path
        image_name (str): Original image name for naming output files
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(image_name).stem
    
    # Save as numpy array for programmatic use
    labels_file = output_dir / f"{base_name}_watershed_labels.npy"
    np.save(labels_file, labels)
    
    # Create visualization: convert labels to colored image
    if np.max(labels) > 0:
        # Generate random colors for each bubble
        num_bubbles = int(np.max(labels))
        colors = np.random.randint(0, 255, size=(num_bubbles + 1, 3))
        colors[0] = [0, 0, 0]  # Background is black
        
        # Create colored visualization
        colored_labels = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for label_id in range(num_bubbles + 1):
            mask = labels == label_id
            colored_labels[mask] = colors[label_id]
        
        # Save visualization
        viz_file = output_dir / f"{base_name}_watershed_labels_viz.png"
        io.imsave(viz_file, colored_labels)
        
        print(f"  Saved watershed labels: {labels_file.name}")
        print(f"  Saved visualization: {viz_file.name}")
    else:
        print(f"  No bubbles detected in {image_name}")

def get_image_shape(image_path):
    """Get the shape of an image without fully loading it."""
    img = io.imread(image_path)
    if len(img.shape) == 3:
        return img.shape[:2]  # (height, width)
    else:
        return img.shape  # Already (height, width)

def test_single_image_progressive():
    """Test progressive quality detection on a single image with watershed conversion."""
    print("=== Testing Progressive Quality Detection with Watershed Conversion ===")
    
    weights_path = r"bubble_analyser/weights/mask_rcnn_bubble.h5"
    input_path = r"tests/sample_images"
    output_path = r"tests/cnn_watershed_result"

    # Find first image in input directory
    input_dir = Path(input_path)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        print("No image files found in input directory")
        return
    
    test_image = image_files[0]
    print(f"Testing with image: {test_image.name}")
    
    # Create detector with default low memory config
    config = BubMaskConfig.for_low_memory_gpu()
    detector = BubMaskDetector(weights_path, config)
    
    # Try progressive detection
    try:
        results = detector.detect_bubbles_progressive(
            test_image,
            return_masks=True,
            return_splash=False
        )
        
        print(f"✓ Success with {results['config_used']} configuration")
        print(f"  Image dimensions used: {results['image_dimensions']}")
        print(f"  Bubbles detected: {results['bubble_count']}")
        
        # Convert to watershed-compatible labels
        if 'masks' in results and results['masks'] is not None:
            # Get original image shape
            original_shape = get_image_shape(test_image)
            
            # Convert BubMask 3D masks to watershed 2D labels
            watershed_labels = convert_bubmask_to_watershed_labels(
                results['masks'], 
                original_shape
            )
            
            print(f"  Converted to watershed labels: shape {watershed_labels.shape}")
            print(f"  Unique labels: {np.unique(watershed_labels)}")
            
            # Save watershed labels
            save_watershed_labels(watershed_labels, output_path, test_image.name)
        else:
            print("  No masks returned from detection")
        
    except Exception as e:
        print(f"✗ Progressive detection failed: {e}")

def test_different_configurations():
    """Test different quality configurations."""
    print("\n=== Testing Different Quality Configurations ===")
    
    weights_path = r"bubble_analyser/weights/mask_rcnn_bubble.h5"
    
    configs = {
        "Low Memory (128x256)": BubMaskConfig.for_low_memory_gpu(),
        "Medium Quality (256x512)": BubMaskConfig.for_medium_quality(),
        "High Quality (512x1024)": BubMaskConfig.for_high_quality(),
    }
    
    for name, config in configs.items():
        try:
            detector = BubMaskDetector(weights_path, config)
            print(f"✓ {name}: Model loaded successfully")
        except Exception as e:
            print(f"✗ {name}: Failed to load - {e}")

def test_batch_detection():
    """Test batch detection with watershed conversion."""
    print("\n=== Testing Batch Detection with Watershed Conversion ===")
    
    weights_path = r"bubble_analyser/weights/mask_rcnn_bubble.h5"
    input_path = r"tests/sample_images"
    output_path = r"tests/cnn_result"
    watershed_output_path = r"tests/cnn_watershed_result"
    
    # Use medium quality for better results while staying memory-safe
    config = BubMaskConfig.for_medium_quality()
    detector = BubMaskDetector(weights_path, config)
    
    try:
        print("Starting batch bubble detection...")
        results_list = detector.batch_detect(
            input_dir=input_path,
            output_dir=output_path,
            save_masks=True,
            save_splash=True
        )
        
        # Process results and convert to watershed format
        total_bubbles = sum(result['bubble_count'] for result in results_list)
        print(f"\n✓ Batch Detection Complete!")
        print(f"  Configuration: Medium Quality (256x512)")
        print(f"  Total images processed: {len(results_list)}")
        print(f"  Total bubbles detected: {total_bubbles}")
        
        print(f"\nConverting to watershed-compatible labels...")
        
        # Convert each result to watershed format
        input_dir = Path(input_path)
        for result in results_list:
            print(f"  Processing {result['image_name']}: {result['bubble_count']} bubbles")
            
            if 'masks' in result and result['masks'] is not None:
                # Find original image to get its shape
                image_files = list(input_dir.glob(f"*{Path(result['image_name']).stem}*"))
                if image_files:
                    original_image = image_files[0]
                    original_shape = get_image_shape(original_image)
                    
                    # Convert to watershed labels
                    watershed_labels = convert_bubmask_to_watershed_labels(
                        result['masks'], 
                        original_shape
                    )
                    
                    # Save watershed labels
                    save_watershed_labels(watershed_labels, watershed_output_path, result['image_name'])
                else:
                    print(f"    Warning: Could not find original image for {result['image_name']}")
            else:
                print(f"    No masks available for {result['image_name']}")
        
        print(f"\n✓ Watershed conversion complete!")
        print(f"  Watershed labels saved to: {watershed_output_path}")
            
    except Exception as e:
        print(f"✗ Batch detection failed: {e}")

def test_watershed_conversion_only():
    """Test only the watershed conversion process on existing BubMask results."""
    print("\n=== Testing Watershed Conversion Only ===")
    
    base_dir = "/Users/eeeyoung/Bubbles/bubble_analyser"
    input_path = os.path.join(base_dir, "tests/sample_images")
    output_path = os.path.join(base_dir, "tests/cnn_watershed_conversion_test")
    weights_path = os.path.join(base_dir, "bubble_analyser/weights/mask_rcnn_bubble.h5")
    
    # Find first image for testing
    input_dir = Path(input_path)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        print("No image files found for conversion test")
        return
    
    test_image = image_files[0]
    print(f"Testing conversion with: {test_image.name}")
    
    try:
        # Quick detection to get masks
        config = BubMaskConfig.for_low_memory_gpu()
        detector = BubMaskDetector(weights_path, config)
        
        results = detector.detect_bubbles(
            test_image,
            return_masks=True,
            return_splash=False
        )
        
        if 'masks' in results and results['masks'] is not None:
            original_shape = get_image_shape(test_image)
            
            print(f"  Original BubMask format: {results['masks'].shape}")
            print(f"  Target image shape: {original_shape}")
            
            # Convert to watershed format
            watershed_labels = convert_bubmask_to_watershed_labels(
                results['masks'], 
                original_shape
            )
            
            print(f"  Converted watershed format: {watershed_labels.shape}")
            print(f"  Unique labels: {len(np.unique(watershed_labels))-1} bubbles + background")
            print(f"  Label range: {np.min(watershed_labels)} to {np.max(watershed_labels)}")
            
            # Save results
            save_watershed_labels(watershed_labels, output_path, test_image.name)
            
            print(watershed_labels)

            print("✓ Watershed conversion test successful!")
        else:
            print("✗ No masks detected for conversion test")
            
    except Exception as e:
        print(f"✗ Watershed conversion test failed: {e}")

def main():
    print("BubMask Progressive Quality Testing with Watershed Conversion")
    print("=" * 70)
    
    # # Test different configurations
    # test_different_configurations()
    
    # Test watershed conversion only
    test_watershed_conversion_only()
    
    # # Test progressive detection on single image with conversion
    # test_single_image_progressive()
    
    # # Test batch detection with conversion
    # test_batch_detection()
    
    print("\n" + "=" * 70)
    print("All testing complete!")
    print("Check the following directories for results:")
    print("  - tests/cnn_result/ (original BubMask outputs)")
    print("  - tests/cnn_watershed_result/ (watershed-compatible labels)")
    print("  - tests/cnn_watershed_conversion_test/ (conversion test results)")

if __name__ == "__main__":
    main()