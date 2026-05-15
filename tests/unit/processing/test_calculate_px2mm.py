import numpy as np
import pytest
from bubble_analyser.processing.calculate_px2mm import resize_to_target_width, get_mm_per_pixel

def test_resize_to_target_width():
    # Create a dummy image (100x200)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    target_width = 100
    resized_img, scale_percent = resize_to_target_width(img, target_width)
    
    assert resized_img.shape[1] == 100
    assert resized_img.shape[0] == 50
    assert scale_percent == 0.5

def test_get_mm_per_pixel():
    pixel_distance = 100.0
    scale_percent = 0.5
    # Original pixel distance = 100 / 0.5 = 200
    # mm_per_pixel = 10.0 / 200 = 0.05
    mm_per_pixel = get_mm_per_pixel(pixel_distance, scale_percent)
    assert mm_per_pixel == 0.05

def test_get_mm_per_pixel_zero():
    # This should probably be handled, but let's see current behavior
    with pytest.raises(ZeroDivisionError):
        get_mm_per_pixel(0, 0.5)
