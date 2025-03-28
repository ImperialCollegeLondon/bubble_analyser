"""This module defines the configuration parameters for the Bubble Analyser project.

The `Config` class is a Pydantic model that validates and manages the configuration
parameters used in the image processing and analysis routines. These parameters
include morphological element sizes, connectivity, marker size, image resampling
factors, and more. The class also includes methods to validate the ranges of these
parameters, ensuring that they are logically consistent before being used in the
processing algorithms.

Classes:
    Config: A Pydantic model for storing and validating configuration parameters.

Methods:
    check_morphological_element_size_range: Validates the morphological element size
    range.
    check_connectivity_range: Validates the connectivity range.
    check_marker_size_range: Validates the marker size range.
    check_resample_range: Validates the resample range.
    check_max_eccentricity_range: Validates the maximum eccentricity range.
    check_min_solidity_range: Validates the minimum solidity range.
    check_min_size_range: Validates the minimum size range.
"""

from pathlib import Path

import typing_extensions
from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictStr,
    model_validator,
)


class Config(BaseModel):  # type: ignore
    """Configuration model for the Bubble Analyser application.

    This class defines and validates all configuration parameters used in the
    image processing and analysis routines. It ensures that all parameters
    are within acceptable ranges and logically consistent.

    Attributes:
        element_size: Size of morphological element for binary operations.
        element_size_range: Valid range for element_size parameter.
        connectivity: Connectivity value (4 or 8) for image processing operations.
        connectivity_range: Valid range for connectivity parameter.
        resample: Factor for resampling images to improve processing speed.
        resample_range: Valid range for resample parameter.
        max_eccentricity: Maximum eccentricity threshold for bubble filtering.
        max_eccentricity_range: Valid range for max_eccentricity parameter.
        min_solidity: Minimum solidity threshold for bubble filtering.
        min_solidity_range: Valid range for min_solidity parameter.
        min_size: Minimum bubble size threshold (in mm).
        min_size_range: Valid range for min_size parameter.
        px2mm: Pixel to millimeter conversion factor.
        bknd_img_path: Path to the background image file.
        threshold_value: Threshold value for watershed segmentation.
        ruler_img_path: Path to the ruler image file for calibration.
        save_path: Path for saving data results and graphs.
        save_path_for_images: Path for saving processed images.
        img_resample: Image resampling factor.
        raw_img_path: Path to the raw input image.
        max_thresh: Maximum threshold value.
        min_thresh: Minimum threshold value.
        step_size: Step size for threshold iteration.
    """

    # Default PARAMETERS
    # ------------------------------Segment Parameters-------------------------------
    # Morphological element used for binary operations, e.g. opening, closing, etc.
    element_size: PositiveInt

    # Connectivity used, use 4 or 8
    connectivity: PositiveInt

    # Images can be resampled to make processing faster
    resample: PositiveFloat
    resample_range: tuple[PositiveFloat, PositiveFloat]

    max_thresh: PositiveFloat
    min_thresh: PositiveFloat
    step_size: PositiveFloat

    high_thresh: PositiveFloat
    mid_thresh: PositiveFloat
    low_thresh: PositiveFloat

    default_range: tuple[StrictFloat, PositiveFloat]

    # User input Image resolution
    px2mm: PositiveFloat

    # Batch processing flag
    do_batch: StrictBool

    # ------------------------------Default Input and Output Settings------------------------------
    # Path for Background image
    bknd_img_path: Path

    # Path for Ruler image
    ruler_img_path: Path

    # Path for saving data results and graphs
    save_path: Path

    # Path for saving images
    save_path_for_images: Path

    # Path for raw image
    raw_img_path: Path

    # ------------------------------Filtering Parameters------------------------------
    # Reject abnormal bubbles from quantification. e.g. E>0.85 or S<0.9
    max_eccentricity: PositiveFloat
    max_eccentricity_range: tuple[PositiveFloat, PositiveFloat]
    min_solidity: PositiveFloat
    min_solidity_range: tuple[PositiveFloat, PositiveFloat]

    # Also ignore too small bubbles (equivalent diameter in mm)
    min_size: PositiveFloat
    min_size_range: tuple[StrictFloat, StrictFloat]

    # Parameters for finding big and small bubbles
    if_find_circles: StrictStr
    L_maxA: PositiveFloat
    L_minA: PositiveFloat
    s_maxA: PositiveFloat
    s_minA: PositiveFloat

    class Config:
        """Pydantic configuration settings for the Config model.

        This nested class configures the behavior of the parent Config model.
        It enables runtime validation of attribute assignments.

        Attributes:
            validate_assignment: When True, validates attributes when they are assigned.
        """

        validate_assignment = True

    @model_validator(mode="after")
    def check_if_within_default_range(self) -> typing_extensions.Self:
        """Validates if the chosen parameter is within default range."""
        if not (self.default_range[0] <= self.max_thresh <= self.default_range[1]):
            raise ValueError("Chosen max_thresh is not within valid range (0, 1)")

        if not (self.default_range[0] <= self.min_thresh <= self.default_range[1]):
            raise ValueError("Chosen min_threshold is not within valid range (0, 1)")

        if not (self.default_range[0] <= self.step_size <= self.default_range[1]):
            raise ValueError("Chosen step_size is not within valid range (0, 1)")

        if not (self.default_range[0] <= self.high_thresh <= self.default_range[1]):
            raise ValueError("Chosen high_thresh is not within valid range (0, 1)")

        if not (self.default_range[0] <= self.mid_thresh <= self.default_range[1]):
            raise ValueError("Chosen mid_thresh is not within valid range (0, 1)")

        if not (self.default_range[0] <= self.low_thresh <= self.default_range[1]):
            raise ValueError("Chosen low_thresh is not within valid range (0, 1)")

        if not (self.default_range[0] <= self.max_eccentricity <= self.default_range[1]):
            raise ValueError("Chosen max_eccentricity is not within valid range (0, 1)")

        if not (self.default_range[0] <= self.min_solidity <= self.default_range[1]):
            raise ValueError("Chosen min_solidity is not within valid range (0, 1)")

        return self

    @model_validator(mode="after")
    def check_morphological_element_size(self) -> typing_extensions.Self:
        """Validates the morphological element size value.

        Ensures that the element_size is one of the allowed values (0, 3, or 5).
        If the value is not allowed, a ValueError is raised.

        Returns:
            Self: The instance itself, for method chaining.

        Raises:
            ValueError: If element_size is not 0, 3, or 5.
        """
        if not (self.element_size == 3 or self.element_size == 5 or self.element_size == 0):
            raise ValueError("Morphological_element_size must be 3, 5 or 0")
        return self

    @model_validator(mode="after")
    def check_connectivity(self) -> typing_extensions.Self:
        """Validates the connectivity value.

        Ensures that the connectivity is one of the allowed values (4 or 8).
        If the value is not allowed, a ValueError is raised.

        Returns:
            Self: The instance itself, for method chaining.
        """
        if not (self.connectivity == 4 or self.connectivity == 8):
            raise ValueError("Connectivity must be 4 or 8")
        return self

    @model_validator(mode="after")
    def check_threshold_order_iterative(self) -> typing_extensions.Self:
        """Validates the threshold order.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order, a ValueError is raised.

        Returns:
            Self: The instance itself, for method chaining.
        """
        low, high = self.min_thresh, self.max_thresh
        if not (high > low):
            raise ValueError("Max threshold must be greater than min threshold\n")
        return self

    @model_validator(mode="after")
    def check_threshold_order_normal(self) -> typing_extensions.Self:
        """Validates the threshold order.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order, a ValueError is raised.

        Returns:
            Self: The instance itself, for method chaining.
        """
        low, mid, high = self.low_thresh, self.mid_thresh, self.high_thresh
        if not (high > mid > low):
            raise ValueError("Values of theshold must be in the order [low < mid < high]\n")
        return self

    @model_validator(mode="after")
    def check_step_size(self) -> typing_extensions.Self:
        """Validates the step size.

        Ensures that the step size is smaller than the difference between max
        thresh and min thresh.

        Returns:
            Self: The instance itself, for method chaining.
        """
        if not (self.step_size < (self.max_thresh - self.min_thresh)):
            raise ValueError("Step size must be smaller than the difference between max and min threshold")
        return self

    @model_validator(mode="after")
    def check_resample_range(self) -> typing_extensions.Self:
        """Validates the resample range.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order (lower bound >= upper bound), a ValueError
        is raised.

        Returns:
            Self: The instance itself, for method chaining.

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.
        """
        # Get the lower and upper bounds of the resample range
        low, high = self.resample_range

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the resample_range are in the wrong order")

        # Return the instance itself for method chaining
        return self

    @model_validator(mode="after")
    def check_max_eccentricity_range(self) -> typing_extensions.Self:
        """Validates the maximum eccentricity range.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order (lower bound >= upper bound), a ValueError
        is raised.

        Returns:
            Self: The instance itself, for method chaining.

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.
        """
        # Get the lower and upper bounds of the maximum eccentricity range
        low, high = self.max_eccentricity_range

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the Max_Eccentricity_range are in the wrong order")

        # Return the instance itself for method chaining
        return self

    @model_validator(mode="after")
    def check_min_solidity_range(self) -> typing_extensions.Self:
        """Validates the minimum solidity range.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order (lower bound >= upper bound), a ValueError
        is raised.

        Returns:
            Self: The instance itself, for method chaining.

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.
        """
        # Get the lower and upper bounds of the minimum solidity range
        low, high = self.min_solidity_range

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the Min_Solidity_range are in the wrong order")

        return self

    @model_validator(mode="after")
    def check_min_size_range(self) -> typing_extensions.Self:
        """Validates the minimum size range.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order (lower bound >= upper bound), a ValueError
        is raised.

        Returns:
            Self: The instance itself, for method chaining.

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.
        """
        # Get the lower and upper bounds of the minimum size range
        low, high = self.min_size_range

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the min_size_range are in the wrong order")
        # Return the instance itself for method chaining
        return self

    @model_validator(mode="after")
    def check_L_area_order(self) -> typing_extensions.Self:
        """Validates the L area order.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order (lower bound >= upper bound), a ValueError
        is raised.

        Returns:
            Self: The instance itself, for method chaining.

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.
        """
        # Get the lower and upper bounds of the minimum size range
        high, low = self.L_maxA, self.L_minA

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the L area range are in the wrong order, should be [L_maxA > L_minA]")
        # Return the instance itself for method chaining
        return self

    @model_validator(mode="after")
    def check_s_area_order(self) -> typing_extensions.Self:
        """Validates the L area order.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order (lower bound >= upper bound), a ValueError
        is raised.

        Returns:
            Self: The instance itself, for method chaining.

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.
        """
        # Get the lower and upper bounds of the minimum size range
        high, low = self.s_maxA, self.s_minA

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the S area range are in the wrong order, should be [s_maxA > s_minA]")
        # Return the instance itself for method chaining
        return self
