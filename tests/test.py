"""Tests for the main module."""

from bubble_analyser import __version__


def test_version():
    """Check that the version is acceptable."""
    assert __version__ == "0.1.0"

### Complete this test file

import unittest
import numpy as np
from pathlib import Path
from bubble_analyser.processing.image_preprocess import image_preprocess
from bubble_analyser.processing.morphological_process import morphological_process

class TestImagePreprocess(unittest.TestCase):

    def test_image_preprocess(self):
        # Define a test image path
        img_path = Path("test_image.JPG")

        # Define a test resampling factor
        img_resample = 0.5

        # Call the image_preprocess function
        img_grey, img_rgb = image_preprocess(img_path, img_resample)

        # Check if the output images are not None
        self.assertIsNotNone(img_grey)
        self.assertIsNotNone(img_rgb)

        # Check if the output images have the correct shape
        self.assertEqual(len(img_grey.shape), 2)  # Grayscale image should have 2 dimensions
        self.assertEqual(len(img_rgb.shape), 3)  # RGB image should have 3 dimensions


class TestMorphologicalProcess(unittest.TestCase):

    def test_morphological_process(self):
        # Define a test binary image
        img_path = Path("test_image.JPG")
        img = io.imread(image_path)
        
        img_binary = np.random.randint(0, 2, size=(512, 512), dtype=np.bool_)

        # Define a test element size
        element_size = 5

        # Call the morphological_process function
        img_processed = morphological_process(img_binary, element_size)

        # Check if the output image is not None
        self.assertIsNotNone(img_processed)

        # Check if the output image has the correct shape
        self.assertEqual(img_processed.shape, img_binary.shape)

        # Check if the output image is of type uint8
        self.assertEqual(img_processed.dtype, np.uint8)
        
if __name__ == "__main__":
    unittest.main()