"""Bubble Analyser: Image Processing for Circular Feature Detection.

This script is designed to process and analyze images to detect and evaluate circular
features, such as bubbles, using various image processing techniques. The script
integrates several modular functions for loading images, processing them, and
calculating properties of detected features in terms of real-world measurements.

The key functionalities include:

1. Loading configuration parameters from a TOML file, which govern the image processing
   steps and parameters.
2. Loading and preprocessing images, including conversion to grayscale and resizing
   based on a given resampling factor.
3. Calculating the conversion factor from pixels to centimeters using a reference ruler
   image, ensuring that measurements of detected features are accurate and scalable.
4. Executing the image processing algorithm, which involves thresholding, morphological
   processing, distance transformation, connected component labeling, and watershed
   segmentation to isolate and analyze circular features.
5. Displaying and saving intermediary and final images, along with calculating and
   printing properties such as equivalent diameter and area of the detected features.

The script is structured to allow easy customization and extension, making it suitable
for a wide range of image analysis tasks that involve circular feature detection and
measurement.

To run the script, simply execute the `default()` function, which orchestrates the
entire process from loading configurations and images to running the analysis and
displaying results.
"""

import toml as tomllib
from pprint import pprint


from .Config import Config
import cv2
import matplotlib.pyplot as plt
import numpy as np
from .calculate_circle_properties import calculate_circle_properties
from .calculate_px2cm import calculate_px2cm
from .image_preprocess import image_preprocess
from .morphological_process import morphological_process
from skimage import (
    color,
    io,
    morphology,
    transform,
)
from .threshold import threshold


def load_image(image_path: str, img_resample: float) -> tuple[np.ndarray, np.ndarray]:
    """Read and preprocess the input image.

    This function loads an image from the specified path, resizes it according to the
    given resampling factor, and converts it to grayscale if the image is in RGB format.

    Args:
        image_path: The file path of the image to load.
        img_resample: The factor by which the image will be resampled (e.g., 0.5 for
        reducing the size by half).

    Returns:
        A tuple containing:
        - The preprocessed grayscale image (if the original was in RGB) or the original
        grayscale image.
        - The resized image in RGB format.
    """
    # Read the input image
    img = io.imread(image_path)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale_percent = img_resample * 100  # percent of original size
    width = int(imgRGB.shape[1] * scale_percent / 100)
    height = int(imgRGB.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    imgRGB = cv2.resize(imgRGB, dim, interpolation=cv2.INTER_AREA)

    img = transform.resize(
        img,
        (int(img.shape[0] * img_resample), int(img.shape[1] * img_resample)),
        anti_aliasing=True,
    )
    if img.ndim > 2:
        img = color.rgb2gray(img)  # Convert to grayscale if the image is in RGB

    return img, imgRGB


def load_toml(file_path: str) -> Config:
    """Load configuration parameters from a TOML file.

    This function reads the TOML configuration file from the specified path and loads
    its contents into a dictionary.

    Args:
        file_path: The file path of the TOML configuration file.

    Returns:
        A dictionary containing the configuration parameters from the TOML file.
    """
    # with open(file_path, "rb") as f:
    #     toml_data: dict[str, str | int | float] = tomllib.load(f)
    #     return toml_data
    with open(file_path, "rb") as f:
        toml_data = tomllib.load(f)

    return Config.Config(**toml_data)


def run_algorithm(
    target_img: np.ndarray,
    bknd_img: np.ndarray,
    imgRGB: np.ndarray,
    params: Config.Config,
    px2cm: float,
    threshold_value: float,
) -> None:
    """Execute the image processing algorithm on the target image.

    This function performs a series of image processing steps on the target image,
    including thresholding, morphological processing, and watershed segmentation.
    It then calculates properties of the detected circular features, such as equivalent
    diameter and area, in centimeters using the provided pixel-to-centimeter ratio.

    Args:
        target_img: The preprocessed target image.
        bknd_img: The background image used for thresholding.
        imgRGB: The resized target image in RGB format.
        params: A dictionary of parameters loaded from the TOML file.
        px2cm: The conversion factor between pixels and centimeters.
        threshold_value: Threshold value for background subtraction

    Returns:
        None. The function displays and saves various intermediary and final images, and
        prints the properties of the detected circular features.
    """
    # Extract parameters from the dictionary
    element_size = morphology.disk(
        params.Morphological_element_size
    )  # Structuring element for morphological operations

    # Below are variables that might be used in the future coding
    # connectivity = params.Connectivity  # Neighborhood connectivity (4 or 8)
    # marker_size = params.Marker_size  # Marker size for watershed segmentation
    # max_eccentricity = params.Max_Eccentricity  # Maximum eccentricity threshold
    # min_solidity = params.Min_Solidity  # Minimum solidity threshold
    # min_bubble_size = params.min_size  # Minimum bubble size (in mm)
    # do_batch = params.do_batch  # Flag for batch processing

    # Display the original image
    plt.figure()
    plt.subplot(231)
    plt.title("1. Original image")
    plt.imshow(target_img, cmap="gray")

    # Apply thresholding and morphological processing
    plt.subplot(232)
    imgThreshold = threshold(target_img, bknd_img, threshold_value)
    imgThreshold = morphological_process(imgThreshold, element_size)
    plt.title("2. Thresh&morph process")
    plt.imshow(imgThreshold * 255, cmap="gray")

    # Apply distance transform
    plt.subplot(233)
    distTrans = cv2.distanceTransform(imgThreshold, cv2.DIST_L2, 5)
    plt.title("3. Distance Transform")
    plt.imshow(distTrans)

    # Apply thresholding to the distance transform
    plt.subplot(234)
    _, distThresh = cv2.threshold(
        distTrans, 0.3 * distTrans.max(), 255, cv2.THRESH_BINARY
    )
    plt.title("4. Threshold of distTrans")
    plt.imshow(distThresh)

    # Apply connected component labeling
    plt.subplot(235)
    distThresh = np.uint8(distThresh)
    _, labels = cv2.connectedComponents(distThresh)
    plt.title("5. Labels")
    plt.imshow(labels)

    # Apply watershed segmentation
    plt.figure()
    plt.subplot(121)
    labels = np.int32(labels)
    labels = cv2.watershed(imgRGB, labels)
    plt.title("6. Final graph after watershed")
    plt.imshow(labels)

    # Display the images
    plt.show()

    # Calculate and print the circle properties
    circle_properties = calculate_circle_properties(labels, px2cm)
    pprint(circle_properties)


def default() -> None:
    """Run the default image processing routine.

    This function loads the configuration parameters from the TOML file, calculates the
    pixel-to-centimeter ratio using a reference ruler image, and then runs the image
    processing algorithm on the target image to detect and analyze circular features.

    Args:
        None.

    Returns:
        None. The function orchestrates the loading of images, execution of the
        algorithm, and display of results.
    """
    # Load parameters from the TOML configuration file
    params = load_toml("./bubble_analyser/config.toml")

    # Read path and image resample factor
    ruler_img_path: str = params.ruler_img_path
    target_img_path: str = params.target_img_path
    bknd_img_path: str = params.background_img_path
    img_resample_factor: float = params.resample
    threshold_value: float = params.threshold_value

    # Calculate the pixel to cm ratio
    px2cm = calculate_px2cm(ruler_img_path, img_resample_factor)
    print(f"Pixel to cm ratio: {px2cm} cm/pixel")

    # Read the background and target image, resize and process into gray scale
    bknd_img, _ = image_preprocess(bknd_img_path, img_resample_factor)
    target_img, imgRGB = image_preprocess(target_img_path, img_resample_factor)

    # Run the default image processing algorithm
    run_algorithm(target_img, bknd_img, imgRGB, params, px2cm, threshold_value)


if __name__ == "__main__":
    default()

# First background subtraction (optional) then otsu thresholding
# Let user define limitations based on the properties of the bubbles for filtering them
# Output the image that eliminate the bubbles being filtered out
# Table and Histogram
# Let user modify the parameters in UI