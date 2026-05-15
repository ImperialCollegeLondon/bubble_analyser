import numpy as np
import pytest
from bubble_analyser.processing.img_sharpening import (
    method_unsharp_single_scale,
    method_unsharp_multi_scale,
    method_clahe_then_usm,
    method_wiener_gaussian,
    method_rl_disk
)

@pytest.fixture
def dummy_rgb():
    # Create a dummy RGB image (50x50)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # Add a white square in the middle
    img[20:30, 20:30, :] = 255
    return img

def test_unsharp_single_scale(dummy_rgb):
    result = method_unsharp_single_scale(dummy_rgb)
    assert result.shape == dummy_rgb.shape
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0

def test_unsharp_multi_scale(dummy_rgb):
    result = method_unsharp_multi_scale(dummy_rgb)
    assert result.shape == dummy_rgb.shape
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0

def test_clahe_then_usm(dummy_rgb):
    result = method_clahe_then_usm(dummy_rgb)
    assert result.shape == dummy_rgb.shape
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0

def test_wiener_gaussian(dummy_rgb):
    result = method_wiener_gaussian(dummy_rgb)
    assert result.shape == dummy_rgb.shape
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0

def test_rl_disk(dummy_rgb):
    result = method_rl_disk(dummy_rgb)
    assert result.shape == dummy_rgb.shape
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0
