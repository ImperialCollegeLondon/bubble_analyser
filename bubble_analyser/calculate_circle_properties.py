import numpy as np
from numpy import typing as npt
from skimage import measure


def calculate_circle_properties(
    labels: npt.NDArray[np.int_], px2cm: float
) -> list[dict[str, float]]:
    """Calculate geometric properties of labeled regions in an image.

    This function computes various properties that describe the "circularity" of regions
    within the labeled image, such as area, equivalent diameter, eccentricity, solidity,
    and circularity. These properties are calculated in centimeters based on the provided
    pixel-to-centimeter ratio.

    Args:
        labels: A labeled image where each distinct region (or "circle") is represented
        by unique labels.
        px2cm: The ratio of centimeters per pixel, used to convert measurements from
        pixels to centimeters.

    Returns:
        A list of dictionaries, each containing the following properties for a region:
        - area: The area of the region in square centimeters.
        - equivalent_diameter: The diameter of a circle with the same area as the region,
        in centimeters.
        - eccentricity: The eccentricity of the ellipse that has the same second-moments
        as the region.
        - solidity: The proportion of the pixels in the convex hull that are also in the
        region.
        - circularity: A measure of how close the shape is to a perfect circle, calculated
        using the perimeter and area.
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
