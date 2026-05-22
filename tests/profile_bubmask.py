"""Debugging BubMask Filtering Performance.

Investigate why the filtering step is extremely slow when using the BubMask algorithm,
profiling the output of BubMask and the `circle_handler` filtering logic.
"""

import cProfile
import logging
import os
import pstats
import sys
from pathlib import Path

# Ensure the root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskConfig, BubMaskDetector
from bubble_analyser.core.services import AnalysisService
from bubble_analyser.processing.circle_handler import FilterParamHandler
from bubble_analyser.processing.config import Config
from bubble_analyser.processing.image import MethodsHandler

logging.basicConfig(level=logging.WARNING)


def main():
    config = Config()
    
    project_root = Path(__file__).resolve().parent.parent
    # Reduce image size or select a specific one
    img_path = project_root / "example_imgs/sample_bubble_images/IMG_9423.JPG"

    # Preload detector
    weights_path = str(project_root / "bubble_analyser/weights/mask_rcnn_bubble.h5")
    if not os.path.exists(weights_path):
        print("Weights not found, skipping profiling")
        sys.exit(0)

    bubmask_config = BubMaskConfig()
    detector = BubMaskDetector(weights_path, bubmask_config)

    service = AnalysisService(config)
    service.algorithm = "BubMask (Deep Learning)"
    service.set_detector(detector)

    # Setup methods handler
    methods_handler = MethodsHandler(config)
    filter_handler = FilterParamHandler(config.model_dump())
    service.setup_methods(methods_handler, filter_handler)

    print("Starting profile...")
    profiler = cProfile.Profile()
    profiler.enable()

    service.process_image(img_path)

    profiler.disable()
    print("Profiling finished")

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
