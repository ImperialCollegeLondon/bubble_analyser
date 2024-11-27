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

import timeit
from pprint import pprint

import cv2
import numpy as np
import toml as tomllib
from numpy import typing as npt
from skimage import (
    color,
    io,
    morphology,
    transform,
)

from .calculate_circle_properties import (
    calculate_circle_properties,
    filter_circle_properties,
)
from .calculate_px2mm import calculate_px2mm
from .config import Config
from .image_postprocess import overlay_labels_on_rgb
from .image_preprocess import image_preprocess
from .morphological_process import morphological_process
from .threshold import threshold


def load_image(
    image_path: str, img_resample: float
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
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
    toml_data = tomllib.load(file_path)

    return Config(**toml_data)


def run_watershed_segmentation(
    target_img: npt.NDArray[np.int_],
    imgRGB: npt.NDArray[np.int_],
    threshold_value: float = 0.3,
    element_size: int = 5,
    connectivity: int = 4,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Run the image processing algorithm on the preprocessed image.

    This function takes the preprocessed image, the original RGB image, the conversion
    factor from millimeters to pixels, and several threshold values as input. It then
    applies watershed segmentation to detect circular features in the image. The
    detected features are then filtered based on their properties, such as eccentricity,
    solidity, circularity, and size.

    The function returns the processed image, the labeled image before filtering, the
    properties of the detected circular features, and the labeled image after filtering.

    Parameters:
        target_img (npt.NDArray[np.int_]): The preprocessed image after thresholding.
        imgRGB (npt.NDArray[np.int_]): The original image in RGB format.
        mm2px (float): The conversion factor from millimeters to pixels.
        threshold_value (float, optional): The threshold value for background subtract.
            Defaults to 0.3.
        element_size (int, optional): The size of the morphological element for binary
            operations. Defaults to 5.
        connectivity (int, optional): The connectivity of the morphological operations.
            Defaults to 4.
        max_eccentricity (float, optional): The maximum eccentricity threshold for
            filtering. Defaults to 1.0.
        min_solidity (float, optional): The minimum solidity threshold for filtering.
            Defaults to 0.9.
        min_circularity (float, optional): The minimum circularity threshold for
            filtering. Defaults to 0.1.
        min_size (float, optional): The minimum size threshold for filtering in pixels.
            Defaults to 0.1.

    Returns:
        tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], list[dict[str, float]],
            npt.NDArray[np.int_]]: A tuple of four arrays, the first being the processed
            image, the second being the labeled image before filtering, the third being
            the properties of the detected circular features, and the fourth being the
            labeled image after filtering.
    """
    start_time = timeit.default_timer()
    distTrans = cv2.distanceTransform(target_img, cv2.DIST_L2, element_size)
    print(f"Distance transform time: {timeit.default_timer() - start_time:.4f} sec")

    start_time = timeit.default_timer()
    # Apply thresholding to the distance transform - sure foreground area
    _, distThresh = cv2.threshold(
        distTrans, threshold_value * distTrans.max(), 255, cv2.THRESH_BINARY
    )
    print(f"Thresholding time: {timeit.default_timer() - start_time:.4f} sec")

    start_time = timeit.default_timer()
    sure_fg_initial = distThresh.copy()
        
    sure_bg = np.array(cv2.dilate(target_img, np.ones((3, 3), np.uint8), iterations=3), dtype=np.uint8)
    sure_fg = np.array(sure_fg_initial, dtype=np.uint8)

    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    print(
        f"Morphological operations time: {timeit.default_timer() - start_time:.4f} sec"
    )

    start_time = timeit.default_timer()
    distThresh = distThresh.astype(np.uint8)
    
    _, labels = cv2.connectedComponents(sure_fg, connectivity) # type: ignore
    labels = labels.astype(np.int32)
    labels = labels + 1
    labels[unknown != 0] = 0
    print(f"Connected components time: {timeit.default_timer() - start_time:.4f} sec")

    start_time = timeit.default_timer()
    labels_watershed = cv2.watershed(imgRGB, labels).astype(np.int_)
    print(f"Watershed time: {timeit.default_timer() - start_time:.4f} sec")

    start_time = timeit.default_timer()
    imgRGB_before_filtering = imgRGB.copy()
    imgRGB_before_filtering = overlay_labels_on_rgb(
        imgRGB_before_filtering, labels_watershed
    )
    print(
        f"Overlay labels before filtering time: {timeit.default_timer() - start_time:.4f} sec"
    )
    return imgRGB_before_filtering, labels_watershed


def final_circles_filtering(
    imgRGB: npt.NDArray[np.int_],
    labels: npt.NDArray[np.int_],
    mm2px: float,
    max_eccentricity: float,
    min_solidity: float,
    min_circularity: float,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], list[dict[str, float]]]:
    """Filter the circles in the image based on their properties.

    Args:
        labels (npt.NDArray[np.int_]): The labels of the circles in the image.
        mm2px (float): The conversion factor from millimeters to pixels.
        max_eccentricity (float): The maximum eccentricity threshold for filtering.
        min_solidity (float): The minimum solidity threshold for filtering.
        min_circularity (float): The minimum circularity threshold for filtering.

    Returns:
        npt.NDArray[np.int_]: The filtered labels of the circles in the image.
    """
    start_time = timeit.default_timer()
    labels = filter_circle_properties(
        labels, mm2px, max_eccentricity, min_solidity, min_circularity
    )
    print(f"Filter properties time: {timeit.default_timer() - start_time:.4f} sec")

    start_time = timeit.default_timer()
    circle_properties = calculate_circle_properties(labels, mm2px)
    print(f"Calculate properties time: {timeit.default_timer() - start_time:.4f} sec")
    pprint(circle_properties)

    start_time = timeit.default_timer()
    imgRGB_overlay = overlay_labels_on_rgb(imgRGB, labels)
    print(f"Overlay labels time: {timeit.default_timer() - start_time:.4f} sec")

    return imgRGB_overlay, labels, circle_properties


def pre_processing() -> (
    tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], Config, float, float]
):
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
    ruler_img_path = params.ruler_img_path
    target_img_path = params.target_img_path
    bknd_img_path = params.background_img_path
    img_resample_factor = params.resample
    threshold_value = params.threshold_value

    # Calculate the pixel to mm ratio
    mm2px, _ = calculate_px2mm(ruler_img_path, img_resample_factor)
    print(f"Pixel to mm ratio: {mm2px} mm/pixel")

    # Read the background and target image, resize and process into gray scale
    bknd_img, _ = image_preprocess(bknd_img_path, img_resample_factor)
    target_img, imgRGB = image_preprocess(target_img_path, img_resample_factor)

    # Apply thresholding and morphological processing
    imgThreshold = threshold(target_img, bknd_img, threshold_value)
    element_size = morphology.disk(params.Morphological_element_size)
    imgThreshold_new = morphological_process(imgThreshold, element_size)

    # plt.figure()
    # plt.subplot(231)
    # plt.title("1. Original image")
    # plt.imshow(target_img, cmap="gray")
    # plt.subplot(232)
    # plt.title("2. Thresh process")
    # plt.imshow(imgThreshold * 255, cmap="gray")
    # plt.subplot(233)
    # plt.title("3. morphological process")
    # plt.imshow(imgThreshold * 255, cmap="gray")
    # plt.show()

    # Run the default image processing algorithm
    return imgThreshold_new, imgRGB, params, mm2px, threshold_value


def main() -> None:
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
    imgThreshold, imgRGB, params, px2mm, threshold_value = pre_processing()
    # Run the default image processing algorithm
    img_overlay, labels_watershed = run_watershed_segmentation(
        imgThreshold,
        imgRGB,
        threshold_value,
        element_size=params.Morphological_element_size,
        connectivity=4,
    )
    imgRGB_overlay, labels, circle_properties = final_circles_filtering(
        imgRGB,
        labels_watershed,
        px2mm,
        max_eccentricity=params.Max_Eccentricity,
        min_solidity=params.Min_Solidity,
        min_circularity=params.Min_Circularity,
    )


if __name__ == "__main__":
    main()
