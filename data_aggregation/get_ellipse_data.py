import logging

import cv2
import numpy as np


class MTImageProcessor:
    """Process binary MT image to extract and analyze ellipse-like shapes."""

    def __init__(self, px2mm: float = 1, resample: float = 1, min_area: int = 50, min_contour_length: int = 10):
        """Initialize the MT Image Processor.

        Args:
            px2mm: Pixel to millimeter conversion factor for display
            resample: Image resampling factor
            min_area: Minimum area (in pixels) for a shape to be considered
            min_contour_length: Minimum contour length for ellipse fitting
        """
        self.px2mm_display = px2mm
        self.resample = resample
        self.real_px2mm = self.px2mm_display * self.resample
        self.mm2px = 1 / self.real_px2mm

        # Filtering parameters
        self.min_area = min_area
        self.min_contour_length = min_contour_length

        self.ellipses = []
        self.ellipse_properties = []

    def process_binary_image(self, binary_image: np.ndarray) -> tuple[list, list[dict]]:
        """Process binary image to extract shapes and calculate their properties.

        Args:
            binary_image: Binary image containing ellipse-like shapes (img_grey_morph_eroded)

        Returns:
            Tuple containing:
            - List of fitted ellipses
            - List of dictionaries with ellipse properties
        """
        # Ensure the image is binary (0 and 255)
        if binary_image.dtype != np.uint8:
            binary_image = binary_image.astype(np.uint8)

        # Apply morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find connected components (individual shapes)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        logging.info(f"Found {num_labels - 1} shapes in the binary image")

        ellipses = []
        valid_shapes = 0
        filtered_shapes = 0

        # Process each connected component (skip label 0 which is background)
        for label in range(1, num_labels):
            # Get area of current component
            area = stats[label, cv2.CC_STAT_AREA]

            # Filter out small noise
            if area < self.min_area:
                filtered_shapes += 1
                continue

            # Create mask for current shape
            mask = np.zeros_like(binary_image, dtype=np.uint8)
            mask[labels == label] = 255

            # Find contours of the current shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Additional filtering: check contour length and area
                contour_area = cv2.contourArea(contour)
                contour_length = len(contour)

                if contour_area < self.min_area or contour_length < self.min_contour_length:
                    continue

                # Need at least 5 points to fit an ellipse
                if contour_length >= 5:
                    try:
                        ellipse = cv2.fitEllipse(contour)

                        # Additional validation: check if ellipse dimensions are reasonable
                        center, axes, angle = ellipse
                        major_axis, minor_axis = max(axes), min(axes)

                        # Filter out degenerate ellipses
                        if major_axis > 0 and minor_axis > 0 and major_axis / minor_axis < 10:
                            ellipses.append(ellipse)
                            valid_shapes += 1
                            logging.debug(
                                f"Fitted ellipse for shape {label}: center={center}, axes={axes}, angle={angle}"
                            )
                        else:
                            logging.debug(
                                f"Filtered degenerate ellipse for shape {label}: axes ratio = {major_axis / minor_axis if minor_axis > 0 else 'inf'}"
                            )

                    except cv2.error as e:
                        logging.warning(f"Could not fit ellipse for shape {label}: {e}")
                        continue
                else:
                    logging.debug(f"Shape {label} has insufficient contour points ({contour_length}) to fit ellipse")

        logging.info(f"Valid ellipses found: {valid_shapes}")
        logging.info(f"Small shapes filtered out: {filtered_shapes}")

        self.ellipses = ellipses

        # Calculate properties for each ellipse
        self.ellipse_properties = self.calculate_ellipse_properties()

        return ellipses, self.ellipse_properties

    def calculate_ellipse_properties(self) -> list[dict[str, float]]:
        """Calculate geometric properties for each detected ellipse.

        Returns:
            List of dictionaries containing ellipse properties
        """
        ellipse_properties = []

        for i, ellipse in enumerate(self.ellipses):
            center, axes, angle = ellipse

            # Get major and minor axis lengths
            major_axis_length = max(axes)
            minor_axis_length = min(axes)

            # Calculate properties
            area = np.pi * (major_axis_length / 2) * (minor_axis_length / 2)
            perimeter = np.pi * (
                3 * (major_axis_length + minor_axis_length)
                - np.sqrt((3 * major_axis_length + minor_axis_length) * (major_axis_length + 3 * minor_axis_length))
            )
            eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            equivalent_diameter = np.sqrt(4 * area / np.pi)

            # Convert to millimeters if needed
            area_mm2 = area * (self.mm2px**2)
            major_axis_mm = major_axis_length * self.mm2px
            minor_axis_mm = minor_axis_length * self.mm2px
            perimeter_mm = perimeter * self.mm2px
            equivalent_diameter_mm = equivalent_diameter * self.mm2px

            properties = {
                "major_axis_length_px": major_axis_length,
                "minor_axis_length_px": minor_axis_length,
                "area_px2": area,
                "perimeter_px": perimeter,
                "eccentricity": eccentricity,
                "equivalent_diameter_px": equivalent_diameter,
            }

            ellipse_properties.append(properties)

        return ellipse_properties

    def visualize_ellipses(self, original_image: np.ndarray, thickness: int = 2) -> np.ndarray:
        """Draw fitted ellipses on the original image for visualization.

        Args:
            original_image: Original image to draw ellipses on
            thickness: Thickness of ellipse outlines

        Returns:
            Image with ellipses drawn
        """
        if len(original_image.shape) == 2:
            # Convert grayscale to RGB for colored ellipses
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = original_image.copy()

        # Draw each ellipse
        for i, ellipse in enumerate(self.ellipses):
            center, axes, angle = ellipse

            # Use different colors for different ellipses
            color = (int(255 * (i % 3) / 2), int(255 * ((i + 1) % 3) / 2), int(255 * ((i + 2) % 3) / 2))

            try:
                cv2.ellipse(vis_image, ellipse, color, thickness)
                # Draw center point
                cv2.circle(vis_image, (int(center[0]), int(center[1])), 3, color, -1)
                # Add shape ID text
                cv2.putText(
                    vis_image,
                    str(i + 1),
                    (int(center[0] + 10), int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            except cv2.error as e:
                logging.warning(f"Could not draw ellipse {i}: {e}")
                continue

        return vis_image

    def save_results(self, output_path: str = "ellipse_properties.csv"):
        """Save ellipse properties to a CSV file.

        Args:
            output_path: Path to save the CSV file
        """
        import pandas as pd

        if self.ellipse_properties:
            df = pd.DataFrame(self.ellipse_properties)
            df.to_csv(output_path, index=False)
            logging.info(f"Ellipse properties saved to {output_path}")
            print(f"Saved {len(self.ellipse_properties)} ellipse properties to {output_path}")
        else:
            logging.warning("No ellipse properties to save")


# Example usage function
def process_mt_image_from_watershed(
    img_grey_morph_eroded: np.ndarray,
    px2mm: float = 1,
    resample: float = 1,
    min_area: int = 50,
    min_contour_length: int = 10,
) -> tuple[list, list[dict]]:
    """Process the MT image (img_grey_morph_eroded) from watershed methods.

    Args:
        img_grey_morph_eroded: Binary image from watershed processing
        px2mm: Pixel to millimeter conversion factor
        resample: Image resampling factor
        min_area: Minimum area for valid shapes
        min_contour_length: Minimum contour length for ellipse fitting

    Returns:
        Tuple containing ellipses and their properties
    """
    processor = MTImageProcessor(
        px2mm=px2mm, resample=resample, min_area=min_area, min_contour_length=min_contour_length
    )
    ellipses, properties = processor.process_binary_image(img_grey_morph_eroded)

    logging.info(f"Processed {len(ellipses)} ellipse-like shapes")

    return ellipses, properties


if __name__ == "__main__":
    # Example of how to use this with your watershed results
    import cv2

    # Configure logging to see info messages
    logging.basicConfig(level=logging.INFO)

    # Load a test binary image (replace with your img_grey_morph_eroded)
    test_image_path = "C:/new_sizer/bubble_analyser/tests/test_image_mt.JPG"
    # test_image_path = "C:/new_sizer/bubble_analyser/tests/IMG_8514_mask.png"

    binary_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    if binary_image is not None:
        # Process the image with filtering parameters
        # Adjust min_area and min_contour_length based on your image characteristics
        processor = MTImageProcessor(px2mm=1, resample=1, min_area=100, min_contour_length=15)
        ellipses, properties = processor.process_binary_image(binary_image)

        # Print results
        print(f"Found {len(ellipses)} valid ellipse-like shapes")
        for prop in properties:
            print(
                f"Shape {prop['shape_id']}: Area = {prop['area_mm2']:.2f} mm², "
                f"Eccentricity = {prop['eccentricity']:.3f}"
            )

        # Visualize results
        vis_image = processor.visualize_ellipses(binary_image)
        cv2.imwrite("ellipses_visualization.jpg", vis_image)

        # Save properties to CSV
        processor.save_results("ellipse_properties.csv")
    else:
        print(f"Could not load image: {test_image_path}")
