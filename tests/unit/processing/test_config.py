import pytest
from pathlib import Path
from bubble_analyser.processing.config import Config

@pytest.fixture
def valid_config_dict():
    return {
        "element_size": 3,
        "connectivity": 8,
        "target_width": 800,
        "target_width_range": (400, 1600),
        "resample": 0.5,
        "resample_range": (0.1, 1.0),
        "max_thresh": 0.9,
        "min_thresh": 0.1,
        "step_size": 0.05,
        "high_thresh": 0.8,
        "mid_thresh": 0.5,
        "low_thresh": 0.2,
        "threshold_value": 0.5,
        "default_range": (0.0, 1.0),
        "if_gaussianblur": "True",
        "ksize": 3,
        "px2mm": 0.1,
        "do_batch": False,
        "bknd_img_path": Path("bknd.png"),
        "ruler_img_path": Path("ruler.png"),
        "save_path": Path("results"),
        "save_path_for_images": Path("results/images"),
        "raw_img_path": Path("raw.png"),
        "max_eccentricity": 0.85,
        "max_eccentricity_range": (0.5, 1.0),
        "min_solidity": 0.9,
        "min_solidity_range": (0.5, 1.0),
        "min_size": 0.1,
        "min_size_range": (0.01, 10.0),
        "max_size": 100.0,
        "if_find_circles": "False",
        "L_maxA": 100.0,
        "L_minA": 10.0,
        "s_maxA": 5.0,
        "s_minA": 1.0,
    }

def test_config_valid(valid_config_dict):
    config = Config(**valid_config_dict)
    assert config.element_size == 3
    assert config.connectivity == 8

def test_config_invalid_element_size(valid_config_dict):
    valid_config_dict["element_size"] = 4
    with pytest.raises(ValueError, match="Morphological_element_size must be 3, 5 or 0"):
        Config(**valid_config_dict)

def test_config_invalid_threshold_order(valid_config_dict):
    valid_config_dict["min_thresh"] = 0.9
    valid_config_dict["max_thresh"] = 0.1
    with pytest.raises(ValueError, match="Max threshold must be greater than min threshold"):
        Config(**valid_config_dict)

def test_config_invalid_gaussianblur(valid_config_dict):
    valid_config_dict["if_gaussianblur"] = "Maybe"
    with pytest.raises(ValueError, match="if_gaussianblur must be 'True' or 'False'"):
        Config(**valid_config_dict)

def test_config_invalid_ksize(valid_config_dict):
    valid_config_dict["ksize"] = 4 # Even number
    with pytest.raises(ValueError, match="ksize must be a positive odd integer"):
        Config(**valid_config_dict)

def test_config_invalid_normal_threshold_order(valid_config_dict):
    valid_config_dict["low_thresh"] = 0.5
    valid_config_dict["mid_thresh"] = 0.2
    valid_config_dict["high_thresh"] = 0.8
    with pytest.raises(ValueError, match="Values of theshold must be in the order"):
        Config(**valid_config_dict)
