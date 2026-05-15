import numpy as np
import pytest
from bubble_analyser.processing.circle_handler import FilterParamHandler, EllipseHandler

@pytest.fixture
def params_dict():
    return {
        "max_eccentricity": 0.8,
        "min_solidity": 0.9,
        "min_size": 1.0,
        "max_size": 100.0,
        "if_find_circles": "N",
        "L_maxA": 50.0,
        "L_minA": 20.0,
        "s_maxA": 10.0,
        "s_minA": 5.0,
    }

def test_filter_param_handler(params_dict):
    handler = FilterParamHandler(params_dict)
    p1, p2 = handler.get_needed_params()
    assert p1["max_eccentricity"] == 0.8
    assert p2["find_circles(Y/N)"] == "N"

def test_ellipse_handler_init():
    labels = np.zeros((100, 100), dtype=np.int_)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    handler = EllipseHandler(labels, img, px2mm_display=10.0, resample=1.0)
    assert handler.real_px2mm == 10.0
    assert handler.mm2px == 0.1

def test_filter_labels_properties(params_dict):
    # Create a dummy labeled image
    labels = np.ones((100, 100), dtype=np.int_)
    # Add a region (label 2) - a small square 10x10
    labels[10:20, 10:20] = 2
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    handler = EllipseHandler(labels, img, px2mm_display=1.0, resample=1.0)
    handler.load_filter_params(params_dict, params_dict) # Just need the keys
    
    # Manually fix params_dict for load_filter_params as it expects internal names
    p1 = {
        "max_eccentricity": 0.8,
        "min_solidity": 0.9,
        "min_size": 1.0,
        "max_size": 1000.0
    }
    p2 = {
        "find_circles(Y/N)": "N",
        "L_maxA": 50.0,
        "L_minA": 20.0,
        "s_maxA": 10.0,
        "s_minA": 5.0,
    }
    handler.load_filter_params(p1, p2)
    
    filtered_labels = handler.filter_labels_properties()
    # Region (label 2) area is 100. mm2px is 1.0. area in mm is 100.
    # It should pass if min_size <= 100 <= max_size
    assert 2 in np.unique(filtered_labels)

def test_calculate_circle_properties():
    handler = EllipseHandler(px2mm_display=1.0, resample=1.0)
    # Circle at (50, 50) with axes (20, 20) -> radius 10
    handler.ellipses = [((50.0, 50.0), (20, 20), 0.0)]
    props = handler.calculate_circle_properties()
    
    assert len(props) == 1
    # mm2px = 1.0. major = 20 * 1.0 = 20. minor = 20 * 1.0 = 20.
    assert props[0]["major_axis_length"] == 20.0
    assert props[0]["minor_axis_length"] == 20.0
    assert pytest.approx(props[0]["area"]) == np.pi * 10 * 10
