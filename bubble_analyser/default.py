import tomllib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import (
    color,
    filters,
    io,
    measure,
    morphology,
    segmentation,
    transform,
)
from skimage.io import imsave


def extendedmin(img, H):
    mask = img.copy()
    marker = mask + H
    hmin = morphology.reconstruction(marker, mask, method="erosion")
    return morphology.local_minima(hmin)


def imposemin(img, minima):
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0
    mask = np.minimum((img + 1), marker)
    return morphology.reconstruction(marker, mask, method="erosion")


def default(image_path, params):
    # Read the input image
    img = io.imread(image_path)

    # Extract parameters from the params dictionary
    se = morphology.disk(
        params["Morphological_element_size"]
    )  # Structuring element for morphological operations
    nb = params["Connectivity"]  # Neighborhood connectivity (4 or 8)
    marker_size = params["Marker_size"]  # Marker size for watershed segmentation
    px2mm = params["px2mm"]  # Image resolution conversion factor (pixels to mm)
    img_resample = params["resample"]  # Resampling factor for image resizing
    bknd_img_path = params["background_img"]  # Path to the background image
    E_max = params["Max_Eccentricity"]  # Maximum eccentricity threshold
    S_min = params["Min_Solidity"]  # Minimum solidity threshold
    Dmin = params["min_size"]  # Minimum bubble size (in mm)
    do_batch = params["do_batch"]  # Flag for batch processing

    # Resize the image for faster processing
    img = transform.resize(
        img,
        (int(img.shape[0] * img_resample), int(img.shape[1] * img_resample)),
        anti_aliasing=True,
    )
    if img.ndim > 2:
        img = color.rgb2gray(img)  # Convert to grayscale if the image is in RGB

    # Use background image for thresholding if provided
    # if bknd_img_path is not None:
    #     bknd_img = io.imread(bknd_img_path)
    #     if bknd_img.ndim > 2:
    #         bknd_img = color.rgb2gray(bknd_img)  # Convert background image to grayscale
    #     T = filters.threshold_local(bknd_img, 3)  # Local threshold for background image+-
    #     imsave('T.png', T.astype(np.uint8) * 255)
    #     BW = img > transform.rescale(T, img_resample)  # Binary image using background threshold
    #     # BW = img > T
    # else:

    BW = img > filters.threshold_otsu(img)  # Binary image using Otsu's thresholding
    imsave("tests/BW_image.png", BW.astype(np.uint8) * 255)

    # Perform morphological closing and fill holes
    B = morphology.closing(~BW, se)
    imsave("tests/BW_closing.png", B.astype(np.uint8) * 255)
    B = ndimage.binary_fill_holes(B)
    imsave("tests/BW_binaryfill.png", B.astype(np.uint8) * 255)
    B = segmentation.clear_border(B)
    imsave("tests/BW_clearborder.png", B.astype(np.uint8) * 255)

    # Use distance transform and watershed for segmentation
    R = -ndimage.distance_transform_edt(~B)
    print(R)
    R2 = morphology.h_minima(-R, marker_size)
    Ld2 = segmentation.watershed(R2, connectivity=nb)
    B[Ld2 == 0] = 0
    imsave("tests/BW_after_R2_watershed.png", B.astype(np.uint8) * 255)

    CH = morphology.convex_hull_image(B)
    R3 = -ndimage.distance_transform_edt(~CH)

    mask = extendedmin(R3, nb)
    R4 = imposemin(R3, mask)
    Ld3 = segmentation.watershed(R4, connectivity=nb)
    CH[Ld3 == 0] = 0

    imsave("tests/R.png", R.astype(np.uint8) * 255)
    imsave("tests/R2.png", R2.astype(np.uint8) * 255)
    imsave("tests/R3.png", R3.astype(np.uint8) * 255)
    imsave("tests/Ld2.png", Ld2.astype(np.uint8) * 255)

    # Fix peak_local_max and watershed usage

    # Label the connected components and calculate geometric properties
    CC = measure.label(CH, connectivity=4)
    props = measure.regionprops(CC, "equivalent_diameter", "eccentricity", "solidity")

    # Extract properties and apply thresholds
    E = np.array([prop.eccentricity for prop in props])
    D = np.array([prop.equivalent_diameter for prop in props])
    S = np.array([prop.solidity for prop in props])
    D = D * px2mm / img_resample

    # Filter out abnormal bubbles
    idx = (E >= E_max) | (S <= S_min) | (D < Dmin)
    D = D[~idx]

    # Collect extra information about bubbles
    extra_info = {"Eccentricity": E[~idx], "Solidity": S[~idx]}

    # Create a labeled image if not doing batch processing
    if do_batch:
        L_image = None
    else:
        allowableAreaIndexes = ~idx
        keeperIndexes = np.where(allowableAreaIndexes)[0]
        keeperBlobsImage = np.isin(measure.label(CH), keeperIndexes + 1)
        keeperBlobsImage = transform.resize(
            keeperBlobsImage, (img.shape[0], img.shape[1]), anti_aliasing=True
        )
        L_image = measure.label(keeperBlobsImage, connectivity=nb)

    return D, L_image, extra_info


def load_toml(file_path) -> dict:
    """Load TOML data from file"""
    with open(file_path, "rb") as f:
        toml_data: dict = tomllib.load(f)
        return toml_data


if __name__ == "__main__":
    # Load parameters from the TOML configuration file
    params = load_toml("./bubble_analyser/config.toml")
    pprint(params)

    # Path to the input image
    image_path = "./tests/sample_images/03.jpg"

    # Run the default image processing algorithm
    D, L_image, extra_info = default(image_path, params)

    # Print the diameters of detected bubbles
    print("Diameters (mm):", D)

    # Display the labeled image if it exists
    if L_image is not None:
        plt.imshow(L_image)
        plt.show()
