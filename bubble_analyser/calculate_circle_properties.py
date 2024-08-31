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
    labels: npt.NDArray[np.int_], px2cm: float
) -> list[dict[str, float]]:
    """Calculate geometric properties of labeled regions in an image.

    This function computes various properties that describe the "circularity" of regions
    within the labeled image, such as area, equivalent diameter, eccentricity, solidity,
    and circularity. These properties are calculated in centimeters based on the
    provided pixel-to-centimeter ratio.

    Args:
        labels: A labeled image where each distinct region (or "circle") is represented
        by unique labels.
        px2cm: The ratio of centimeters per pixel, used to convert measurements from
        pixels to centimeters.

    Returns:
        A list of dictionaries, each containing the following properties for a region:
        - area: The area of the region in square centimeters.
        - equivalent_diameter: The diameter of a circle with the same area as the region
        , in centimeters.
        - eccentricity: The eccentricity of the ellipse that has the same second-moments
        as the region.
        - solidity: The proportion of the pixels in the convex hull that are also in the
        region.
        - circularity: A measure of how close the shape is to a perfect circle,
        calculated using the perimeter and area.
    """
    properties = measure.regionprops(labels)
    circle_properties = []
    for prop in properties:
        area = prop.area * (px2cm**2)
        equivalent_diameter = prop.equivalent_diameter * px2cm
        eccentricity = prop.eccentricity
        solidity = prop.solidity
        circularity = (4 * np.pi * area) / (prop.perimeter * px2cm) ** 2
        circle_properties.append(
            {
                "area": area,
                "equivalent_diameter": equivalent_diameter,
                "eccentricity": eccentricity,
                "solidity": solidity,
                "circularity": circularity,
            }
        )
    return circle_properties
