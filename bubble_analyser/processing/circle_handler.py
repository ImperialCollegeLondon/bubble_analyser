from collections.abc import Sequence

import cv2
import numpy as np
from numpy import typing as npt
from skimage import measure


class CircleHandler:
    def __init__(
        self,
        labels_before_filtering: npt.NDArray[np.int_],
        img_rgb: npt.NDArray[np.int_],
        px2mm: float = 90.0,
    ) -> None:
        self.filter_param_dict: dict[str, float] = {
            "max_eccentricity": 0.0,
            "min_solidity": 0.0,
            "min_size": 0.0,
        }

        self.img_rgb: npt.NDArray[np.int_] = img_rgb
        self.labels_before_filtering: npt.NDArray[np.int_] = labels_before_filtering
        self.labels_after_filtering: npt.NDArray[np.int_]
        self.labels_for_calculations: npt.NDArray[np.int_]

        self.px2mm: float = px2mm

        self.ellipses: list[tuple[tuple[float, float], tuple[int, int], int]]
        self.ellipses_on_image: npt.NDArray[np.int_]
        self.ellipses_properties: list[dict[str, float]]

    def load_filter_params(self, filter_param_dict: dict[str, float]) -> None:
        self.filter_param_dict = filter_param_dict

    def filter_labels_properties(self) -> npt.NDArray[np.int_]:
        """Filters out regions (circles) from the labeled image based on their properties.

        Args:
            labels: A labeled image where each distinct region is represented by a unique
            label.
            px2mm: The pixel-to-mm conversion factor.
            min_eccentricity: The minimum allowed eccentricity for circles.
            min_solidity: The minimum allowed solidity for circles.
            min_circularity: The minimum allowed circularity for circles.

        Returns:
            Updated labels array where regions not meeting the thresholds are removed.
        """
        labels = self.labels_before_filtering
        px2mm = self.px2mm

        properties = measure.regionprops(labels)
        new_labels = np.copy(labels)
        mm2px = 1 / px2mm

        max_eccentricity = self.filter_param_dict["max_eccentricity"]
        min_solidity = self.filter_param_dict["min_solidity"]
        min_size = self.filter_param_dict["min_size"]

        for prop in properties:
            if prop.label == 1:  # Ignore the background
                continue

            # Calculate circle properties in mm
            area = prop.area * (mm2px**2)
            eccentricity = prop.eccentricity
            solidity = prop.solidity

            # Check if the circle properties meet the thresholds
            if not (
                eccentricity <= max_eccentricity
                and min_solidity <= solidity
                and area >= min_size
            ):
                # Remove the region by setting it to 1 (background)
                new_labels[new_labels == prop.label] = 1

        self.labels_after_filtering = new_labels
        return new_labels

    def fill_ellipse_labels(
        self,
    ) -> list[tuple[tuple[float, float], tuple[int, int], int]]:
        """Fill each ellipse label in labels_before_filtering and return a new label object.

        Returns:
            npt.NDArray[np.int_]: A new label object with filled ellipses.
        """
        # Store ellipses
        ellipses = []

        # Create an empty image to draw ellipses
        for label in np.unique(self.labels_after_filtering):
            if label == 0:
                continue  # Skip the background label

            mask = np.zeros_like(self.labels_after_filtering, dtype=np.uint8)
            mask[self.labels_after_filtering == label] = 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    ellipses.append(ellipse)

        self.ellipses = ellipses
        return ellipses

    def overlay_ellipses_on_image(self, thickness: int = 20) -> npt.NDArray[np.int_]:
        ellipse_image = self.img_rgb

        for ellipse in self.ellipses:
            cv2.ellipse(ellipse_image, ellipse, (0, 0, 255), thickness)
        self.ellipses_on_image = ellipse_image

        self.create_labelled_image_from_ellipses()

        return ellipse_image

    def create_labelled_image_from_ellipses(self) -> npt.NDArray[np.int_]:
        """Creates a labelled image based on the ellipses fitted.
        The resulting image is a 2D array with the same height and width as self.img_rgb where:
          - Background pixels have a value of 1.
          - Each ellipse is filled with a unique label (starting from 2).

        Returns:
            A labelled image as a numpy array of type np.int_.
        """
        height, width = self.img_rgb.shape[:2]

        # Initialize the labelled image with background label (1)
        labelled_img = np.ones((height, width), dtype=np.int_)

        current_label = 2  # Start labelling from 2
        for ellipse in self.ellipses:
            # Create a mask for the current ellipse.
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(mask, ellipse, color=255, thickness=-1)
            # Assign the current label to all pixels inside the ellipse.
            labelled_img[mask == 255] = current_label
            current_label += 1

        cv2.imwrite("outputlabelled_img.png", labelled_img.astype(np.uint8) * 255)
        print("labelled image created")
        return labelled_img

    def calculate_circle_properties(self) -> list[dict[str, float | Sequence[float]]]:
        px2mm = self.px2mm

        ellipse_properties = []
        mm2px = 1 / px2mm

        for ellipse in self.ellipses:
            center, axes, angle = ellipse
            major_axis_length = max(axes) * mm2px
            minor_axis_length = min(axes) * mm2px
            area = np.pi * (major_axis_length / 2) * (minor_axis_length / 2)
            perimeter = np.pi * (
                3 * (major_axis_length + minor_axis_length)
                - np.sqrt(
                    (3 * major_axis_length + minor_axis_length)
                    * (major_axis_length + 3 * minor_axis_length)
                )
            )
            eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            ellipse_properties.append(
                {
                    "major_axis_length": major_axis_length,
                    "minor_axis_length": minor_axis_length,
                    "equivalent_diameter": equivalent_diameter,
                    "area": area,
                    "perimeter": perimeter,
                    "eccentricity": eccentricity,
                }
            )

        self.ellipses_properties = ellipse_properties
        # print(ellipse_properties)
        return ellipse_properties


if __name__ == "__main__":
    input_labels_path = "../../tests/test_labels_watershed.npy"
    output_filled_labels_path = "../../tests/test_labels_after_fill.npy"
    output_filled_labels_img_path = "../../tests/test_labels_after_fill.JPG"

    img_path = "../../tests/test_image_rgb.JPG"
    img_rgb: npt.NDArray[np.int_] = cv2.imread(img_path)  # type: ignore

    # Load the input label object
    labels_before_filtering = np.load(input_labels_path)

    # Create an instance of CircleHandler
    circle_handler = CircleHandler(labels_before_filtering, img_rgb)
    ellipse_image_path = "../../tests/test_image_rgb.JPG"
    ellipse_image: npt.NDArray[np.int_] = cv2.imread(ellipse_image_path)  # type: ignore
    circle_handler.img_rgb = ellipse_image  # type: ignore

    # Fill ellipse labels
    _ = circle_handler.fill_ellipse_labels()
    filled_labels = circle_handler.overlay_ellipses_on_image()
    circle_handler.calculate_circle_properties()
    # Save the filled labels
    np.save(output_filled_labels_path, filled_labels)
    cv2.imwrite(output_filled_labels_img_path, filled_labels)
    print(f"Filled labels saved to {output_filled_labels_path}")
