import numpy as np
import pytest
import cv2
from bubble_analyser.methods.watershed_methods import IterativeWatershed, NormalWatershed

@pytest.fixture
def dummy_images():
    # Create a 100x100 grey image with some "bubbles"
    # Dark bubbles on light background
    img_grey = np.full((100, 100), 200, dtype=np.uint8)
    # Bubble 1
    cv2.circle(img_grey, (30, 30), 10, 50, -1)
    # Bubble 2
    cv2.circle(img_grey, (70, 70), 15, 70, -1)
    
    img_rgb = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2RGB)
    return img_grey, img_rgb

@pytest.fixture
def iterative_params():
    return {
        "resample": 1.0,
        "element_size": 3,
        "connectivity": 8,
        "max_thresh": 0.9,
        "min_thresh": 0.1,
        "step_size": 0.1,
    }

@pytest.fixture
def normal_params():
    return {
        "resample": 1.0,
        "high_thresh": 0.8,
        "mid_thresh": 0.5,
        "low_thresh": 0.2,
        "element_size": 3,
        "connectivity": 8,
    }

def test_iterative_watershed(dummy_images, iterative_params):
    img_grey, img_rgb = dummy_images
    method = IterativeWatershed(iterative_params)
    method.initialize_processing(iterative_params, img_grey, img_rgb, if_bknd_img=False)
    
    labels_on_img, labels_watershed, _ = method.get_results_img()
    
    assert labels_on_img.shape == img_rgb.shape
    assert labels_watershed.shape == img_grey.shape
    # Should find at least 2 bubbles (labels > 1)
    assert len(np.unique(labels_watershed)) >= 3 # Background is usually 1, and some 0s/ -1s from watershed

def test_normal_watershed(dummy_images, normal_params):
    img_grey, img_rgb = dummy_images
    method = NormalWatershed(normal_params)
    method.initialize_processing(normal_params, img_grey, img_rgb, if_bknd_img=False)
    
    labels_on_img, labels_watershed, _ = method.get_results_img()
    
    assert labels_on_img.shape == img_rgb.shape
    assert labels_watershed.shape == img_grey.shape
    assert len(np.unique(labels_watershed)) >= 3
