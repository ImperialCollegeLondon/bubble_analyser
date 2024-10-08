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

from pprint import pprint
import cv2
import matplotlib.pyplot as plt
import numpy as np
import toml as tomllib
from numpy import typing as npt
from skimage import (
    color,
    io,
    morphology,
    transform,
)

from .calculate_circle_properties import calculate_circle_properties, filter_circle_properties
from .calculate_px2mm import calculate_px2mm
from .config import Config
from .image_preprocess import image_preprocess
from .image_postprocess import overlay_labels_on_rgb
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
    mm2px: float,
    threshold_value: float = 0.3,
    element_size: int = 5,
    connectivity: int = 4,
    max_eccentricity: float = 1.0,
    min_solidity: float = 0.9,
    min_circularity: float = 0.1,
    min_size: float = 0.1
) -> tuple[npt.NDArray[np.int_], 
           npt.NDArray[np.int_],
           list[dict[str, float]],
           npt.NDArray[np.int_]]:
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
    # element_size = morphology.disk(
    #     params.Morphological_element_size)
    # Structuring element for morphological operations

    # Below are variables that might be used in the future coding
    # connectivity = params.Connectivity  # Neighborhood connectivity (4 or 8)
    # marker_size = params.Marker_size  # Marker size for watershed segmentation
    # max_eccentricity = params.Max_Eccentricity  # Maximum eccentricity threshold
    # min_solidity = params.Min_Solidity  # Minimum solidity threshold
    # min_bubble_size = params.min_size  # Minimum bubble size (in mm)
    # do_batch = params.do_batch  # Flag for batch processing

    # Display the original image
    distTrans = cv2.distanceTransform(target_img, cv2.DIST_L2, element_size)
    
    # Apply thresholding to the distance transform - sure foreground area
    _, distThresh = cv2.threshold(
        distTrans, threshold_value * distTrans.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = distThresh.copy()
    
    # sure background area
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(target_img, kernel, iterations=3)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    
    # Apply connected component labeling
    distThresh = distThresh.astype(np.uint8)
    _, labels = cv2.connectedComponents(sure_fg, connectivity)
    labels = labels.astype(np.int32)
    labels = labels + 1
    labels[unknown != 0] = 0
    labels_with_unknown = labels.copy()
    
    # Apply watershed segmentation
    labels_final = cv2.watershed(imgRGB, labels).astype(np.int_)
    imgRGB_before_filtering = imgRGB.copy()
    imgRGB_before_filtering = overlay_labels_on_rgb(imgRGB_before_filtering, labels_final)

    # Display the images
    # First display window
    # plt.figure()
    # plt.subplot(331)
    # plt.title("3. Distance Transform")
    # plt.imshow(distTrans)
    # plt.subplot(332)
    # plt.title("4. Sure_fg (distThresh)")
    # plt.imshow(sure_fg)
    # plt.subplot(333)
    # plt.title("5. Sure_bg")
    # plt.imshow(sure_bg)
    # plt.subplot(334)
    # plt.title("6. Unknown")
    # plt.imshow(unknown)
    # plt.subplot(335)
    # plt.title("5. Labels")
    # plt.imshow(labels)
    # plt.subplot(336)
    # plt.title("6. Labels with unknown")
    # plt.imshow(labels_with_unknown)
    # plt.show()
    # plt.subplot(337)
    # plt.title("7. imgRGB with labels")
    # plt.imshow(imgRGB_before_filtering)

    print("max ecccentricity from default:", max_eccentricity)
    # Filter out the bubbles according to threshold of properties of being a circle
    labels_final = filter_circle_properties(labels_final, 
                                            mm2px,
                                            max_eccentricity,
                                            min_solidity,
                                            min_circularity)
    
    circle_properties = calculate_circle_properties(labels_final, mm2px)
    pprint(circle_properties)
     
    imgRGB_overlay = overlay_labels_on_rgb(imgRGB, labels_final)

    # plt.subplot(338)
    # plt.title("8. Labels after filtering")
    # plt.imshow(labels_final)
    # plt.subplot(339)  
    # plt.title("9. Final overlayed image")
    # plt.imshow(imgRGB_overlay)
    # plt.show()

    return imgRGB_overlay, imgRGB_before_filtering, circle_properties, labels_final
    
def pre_processing() -> tuple[npt.NDArray[np.int_], 
                              npt.NDArray[np.int_], 
                              Config, 
                              float, 
                              float]:
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
    element_size = morphology.disk(
        params.Morphological_element_size
    ) 
    imgThreshold = morphological_process(imgThreshold, element_size)

    plt.figure()
    plt.subplot(231)
    plt.title("1. Original image")
    plt.imshow(target_img, cmap="gray")
    plt.subplot(232)
    plt.title("2. Thresh process")
    plt.imshow(imgThreshold * 255, cmap="gray")
    plt.subplot(233)
    plt.title("3. morphological process")
    plt.imshow(imgThreshold * 255, cmap="gray")
    plt.show()
    
    # Run the default image processing algorithm
    return imgThreshold, imgRGB, params, mm2px, threshold_value

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
    img_overlay, circle_properties, labels = run_watershed_segmentation(imgThreshold, 
                                                           imgRGB, 
                                                           px2mm, 
                                                           threshold_value,
                                                           element_size=params.Morphological_element_size,
                                                           connectivity=4)
    
if __name__ == "__main__":
    main()

# First background subtraction (optional) then otsu thresholding - DONE
# Let user define limitations based on the properties of the bubbles for filtering them
# Output the image that eliminate the bubbles being filtered out 
# Table and Histogram 
# Let user modify the parameters in UI - DONE
# Merge default branch - DONE

# Manual input for bubble detection, e.g. img_resample, mask_size, element_size, etc.
# Then filtering.