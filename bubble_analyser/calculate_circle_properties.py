"""Calculate Circle Properties.

This module contains functions for calculating geometric properties of regions
identified in an image. It is particularly focused on regions that are labeled in terms
of their circularity attributes.

The `calculate_circle_properties` function evaluates the geometric features of labeled
regions within an image, which have been identified as separate entities, often through
a segmentation process. It measures various properties related to the shape and size of
the regions, adjusted to real-world dimensions using a provided pixel-to-centimeter
conversion ratio.

Function:
    calculate_circle_properties(labels, px2cm): Computes area, equivalent diameter,
    eccentricity, solidity, and circularity for each labeled region.

Each computed property is defined as follows:
- Area: Total area of the region converted from pixels to square centimeters.
- Equivalent diameter: Diameter of a circle with the equivalent area as the region,
  provided in centimeters.
- Eccentricity: Measure of the deviation of the region from a perfect circle, where
  0 indicates a perfect circle and values closer to 1 indicate elongated shapes.
- Solidity: Ratio of the region's area to the area of its convex hull, indicating
  the compactness of the shape.
- Circularity: A value that describes how closely the shape of the region approaches
  that of a perfect circle, calculated from the area and the perimeter.

The function returns a list of dictionaries, with each dictionary holding the properties
for a specific region, facilitating easy access and manipulation of these metrics in
subsequent analysis or reporting stages.
"""

import numpy as np
from numpy import typing as npt
from skimage import measure


def calculate_circle_properties(
    labels: npt.NDArray[np.int_], mm2px: float
) -> list[dict[str, float]]:
    """Calculate geometric properties of regions identified in an image.

    Parameters:
        labels (npt.NDArray[np.int_]): A labeled image where each distinct region is
            represented by a unique label.
        mm2px (float): The conversion factor from millimeters to pixels.

    Returns:
        list[dict[str, float]]: A list of dictionaries containing the properties of
            each region, including area, equivalent diameter, eccentricity, solidity,
            circularity, and surface diameter. The area is given in square millimeters,
            while the diameters are given in millimeters.
    """
    properties = measure.regionprops(labels)
    circle_properties = []
    for prop in properties:
        if prop.label == 1:  # Ignore the background, labeled as 1
            continue

        area = prop.area * (mm2px**2)
        equivalent_diameter = prop.equivalent_diameter * mm2px
        eccentricity = prop.eccentricity
        solidity = prop.solidity
        circularity = (4 * np.pi * area) / (prop.perimeter * mm2px) ** 2
        surface_diameter = 2 * np.sqrt(area / np.pi)
        circle_properties.append(
            {
                "area": area,
                "equivalent_diameter": equivalent_diameter,
                "eccentricity": eccentricity,
                "solidity": solidity,
                "circularity": circularity,
                "surface_diameter": surface_diameter,
            }
        )
    return circle_properties


def filter_circle_properties(
    labels: npt.NDArray[np.int_],
    px2mm: float,
    max_eccentricity: float = 1.0,
    min_solidity: float = 0.9,
    min_circularity: float = 0.1,
) -> npt.NDArray[np.int_]:
    """Filters out regions (circles) from the labeled image based on their properties.

    Args:
        labels: A labeled image where each distinct region is represented by a unique
        label.
        px2mm: The pixel-to-mm conversion factor.
        min_eccentricity: The minimum allowed eccentricity for circles.
        max_eccentricity: The maximum allowed eccentricity for circles.
        min_solidity: The minimum allowed solidity for circles.
        max_solidity: The maximum allowed solidity for circles.
        min_circularity: The minimum allowed circularity for circles.
        max_circularity: The maximum allowed circularity for circles.

    Returns:
        Updated labels array where regions not meeting the thresholds are removed.
    """
    properties = measure.regionprops(labels)
    new_labels = np.copy(labels)

    for prop in properties:
        if prop.label == 1:  # Ignore the background
            continue

        # Calculate circle properties in mm
        area = prop.area * (px2mm**2)
        # equivalent_diameter = prop.equivalent_diameter * px2mm
        eccentricity = prop.eccentricity
        solidity = prop.solidity
        circularity = (4 * np.pi * area) / (prop.perimeter * px2mm) ** 2

        # Check if the circle properties meet the thresholds
        if not (
            eccentricity <= max_eccentricity
            and min_solidity <= solidity
            and min_circularity <= circularity
        ):
            # Remove the region by setting it to 1 (background)
            new_labels[new_labels == prop.label] = 1

    return new_labels
