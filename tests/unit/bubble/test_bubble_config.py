import pytest
import numpy as np
from bubble_analyser.bubble.bubble import BubbleConfig, BubbleInferenceConfig

def test_bubble_config():
    config = BubbleConfig()
    assert config.NAME == "bubble"
    assert config.NUM_CLASSES == 2 # Background + bubble
    assert config.IMAGE_RESIZE_MODE == "square"
    assert config.GPU_COUNT == 1

def test_bubble_inference_config():
    config = BubbleInferenceConfig()
    assert config.GPU_COUNT == 1
    assert config.IMAGES_PER_GPU == 1
    assert config.IMAGE_RESIZE_MODE == "pad64"
    assert config.DETECTION_MIN_CONFIDENCE == 0.5
