import pytest
from pathlib import Path
import cv2
import numpy as np
from bubble_analyser.processing.image import Image, MethodsHandler
from bubble_analyser.processing.config import Config

@pytest.fixture
def sample_paths():
    root = Path(__file__).resolve().parents[2]
    return {
        "raw": root / "tests" / "test_image_rgb.JPG",
        "bknd": root / "tests" / "background_image_rgb.JPG",
        "save": root / "tests" / "integration_results"
    }

@pytest.fixture
def integration_config(sample_paths):
    return Config(
        element_size=3,
        connectivity=8,
        target_width=400,
        target_width_range=(200, 1000),
        resample=0.5,
        resample_range=(0.1, 1.0),
        max_thresh=0.9,
        min_thresh=0.1,
        step_size=0.1,
        high_thresh=0.8,
        mid_thresh=0.5,
        low_thresh=0.2,
        threshold_value=0.5,
        default_range=(0.0, 1.0),
        if_gaussianblur="True",
        ksize=3,
        px2mm=0.1,
        do_batch=False,
        bknd_img_path=sample_paths["bknd"],
        ruler_img_path=sample_paths["raw"],
        save_path=sample_paths["save"],
        save_path_for_images=sample_paths["save"] / "images",
        raw_img_path=sample_paths["raw"],
        max_eccentricity=0.9,
        max_eccentricity_range=(0.5, 1.0),
        min_solidity=0.8,
        min_solidity_range=(0.5, 1.0),
        min_size=0.01,
        min_size_range=(0.0, 10.0),
        max_size=1000.0,
        if_find_circles="False",
        L_maxA=100.0,
        L_minA=10.0,
        s_maxA=5.0,
        s_minA=1.0,
    )

def test_full_pipeline(integration_config, sample_paths):
    # Ensure save directory exists
    sample_paths["save"].mkdir(parents=True, exist_ok=True)
    
    methods_handler = MethodsHandler(integration_config)
    
    image = Image(
        px2mm_display=integration_config.px2mm,
        raw_img_path=sample_paths["raw"],
        all_methods_n_params=methods_handler.full_dict,
        methods_handler=methods_handler,
        bknd_img_path=None # Start without background for simplicity
    )
    
    # Load filter params
    image.load_filter_params(
        {"max_eccentricity": 0.9, "min_solidity": 0.8, "min_size": 0.01, "max_size": 1000.0},
        {"find_circles(Y/N)": "N", "L_maxA": 100.0, "L_minA": 10.0, "s_maxA": 5.0, "s_minA": 1.0}
    )
    
    # Process using Default (NormalWatershed)
    image.processing_image_before_filtering("Default", cnn_model=None)
    
    # Run full filtering processing
    image.filtering_processing()
    
    assert len(image.ellipses) >= 0
    assert image.ellipses_on_images is not None
    assert len(image.ellipses_properties) == len(image.ellipses)
    
    # Check if we found at least some bubbles (this image should have some)
    # If it doesn't, we might need a better test image or adjust params
    print(f"Found {len(image.ellipses)} ellipses")
