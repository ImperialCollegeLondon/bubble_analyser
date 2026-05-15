import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from bubble_analyser.methods.bubmask_method import BubMaskWatershed

@pytest.fixture
def dummy_images():
    img_grey = np.zeros((100, 100), dtype=np.uint8)
    img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    return img_grey, img_rgb

@pytest.fixture
def bubmask_params():
    return {
        "weights_path": "mock_weights.h5",
        "confidence_threshold": 0.5,
        "resample": 1.0,
        "image_min_dim": 128,
        "image_max_dim": 256,
        "alpha": 0.5,
    }

def test_bubmask_watershed_init(bubmask_params):
    with patch('bubble_analyser.methods.bubmask_method.BubMaskDetector'):
        method = BubMaskWatershed(bubmask_params)
        assert method.name == "BubMask (Deep Learning)"
        assert method.confidence_threshold == 0.5

def test_bubmask_watershed_detect(dummy_images, bubmask_params):
    img_grey, img_rgb = dummy_images
    
    # Mock detector
    mock_detector = MagicMock()
    mock_masks = np.zeros((100, 100, 1), dtype=bool)
    mock_masks[10:20, 10:20, 0] = True
    mock_detector.detect_bubbles.return_value = {
        'masks': mock_masks,
        'bubble_count': 1,
        'scores': np.array([0.9]),
        'rois': np.array([[10, 10, 20, 20]])
    }
    
    with patch('skimage.io.imsave'): # Don't actually save
        method = BubMaskWatershed(bubmask_params)
        method.initialize_processing(bubmask_params, img_grey, img_rgb, False, cnn_model=mock_detector)
        
        labels_on_img, labels_watershed, _ = method.get_results_img()
        
        assert labels_on_img.shape == img_rgb.shape
        assert labels_watershed.shape == img_grey.shape
        assert 2 in np.unique(labels_watershed) # Label 2 for the first bubble
