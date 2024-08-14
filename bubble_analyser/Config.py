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
    model_validator,
)


class Config(BaseModel):
    """A Pydantic model for storing and validating configuration parameters.

    The class contains parameters for image processing and analysis, such as
    morphological element sizes, connectivity, marker size, image resampling
    factors, maximum eccentricity, and more. The class also includes methods to
    validate the ranges of these parameters, ensuring that they are logically
    consistent before being used in the processing algorithms.
    # Morphological element used for binary operations, e.g. opening, closing, etc.
    Morphological_element_size: PositiveInt
    Morphological_element_size_range: tuple[PositiveInt, PositiveInt]

    Attributes:
        Morphological_element_size: PositiveInt
        Morphological_element_size_range: tuple[PositiveInt, PositiveInt]
        Connectivity: PositiveInt
        Connectivity_range: tuple[PositiveInt, PositiveInt]
        Marker_size: PositiveInt
        Marker_size_range: tuple[PositiveInt, PositiveInt]
        resample: PositiveFloat
        resample_range: tuple[PositiveFloat, PositiveFloat]
        Max_Eccentricity: PositiveFloat
        Max_Eccentricity_range: tuple[PositiveFloat, PositiveFloat]
        Min_Solidity: PositiveFloat
        Min_Solidity_range: tuple[PositiveFloat, PositiveFloat]
        min_size: StrictFloat
        min_size_range: tuple[StrictFloat, StrictFloat]
        px2mm: PositiveFloat
        target_img_path: Path
        background_img_path: Path
        threshold_value: PositiveFloat
        ruler_img_path: Path
        do_batch: StrictBool
    """

    # Default PARAMETERS

    # Morphological element used for binary operations, e.g. opening, closing, etc.
    Morphological_element_size: PositiveInt
    Morphological_element_size_range: tuple[PositiveInt, PositiveInt]

    # Connectivity used, use 4 or 8
    Connectivity: PositiveInt
    Connectivity_range: tuple[PositiveInt, PositiveInt]

    # Marker size for watershed segmentation
    Marker_size: PositiveInt
    Marker_size_range: tuple[PositiveInt, PositiveInt]

    # Images can be resampled to make processing faster
    resample: PositiveFloat
    resample_range: tuple[PositiveFloat, PositiveFloat]

    # Reject abnormal bubbles from quantification. E>0.85 or S<0.9
    Max_Eccentricity: PositiveFloat
    Max_Eccentricity_range: tuple[PositiveFloat, PositiveFloat]
    Min_Solidity: PositiveFloat
    Min_Solidity_range: tuple[PositiveFloat, PositiveFloat]

    # Also ignore too small bubbles (equivalent diameter in mm)
    min_size: StrictFloat
    min_size_range: tuple[StrictFloat, StrictFloat]

    # User input Image resolution
    px2mm: PositiveFloat

    # Path for Target image
    target_img_path: Path

    # Path for Background image
    background_img_path: Path

    # Threshold value for background subtraction
    threshold_value: PositiveFloat

    # Path for Ruler image
    ruler_img_path: Path

    # Batch processing flag
    do_batch: StrictBool

    @model_validator(mode="after")
    def check_morphological_element_size_range(self) -> typing_extensions.Self:
        """Validates the morphological element size range.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order, a ValueError is raised.

        Returns:
            Self: The instance itself, for method chaining.
        """
        low, high = self.Morphological_element_size_range
        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError(
                "Limits for the Morphological_element_size_range are in the wrong order"
            )
        return self

    @model_validator(mode="after")
    def check_connectivity_range(self) -> typing_extensions.Self:
        """Validates the connectivity range.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order, a ValueError is raised.

        Returns:
            Self: The instance itself, for method chaining.
        """
        low, high = self.Connectivity_range
        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the Connectivity_range are in the wrong order")
        return self

    @model_validator(mode="after")
    def check_marker_size_range(self) -> typing_extensions.Self:
        """Validates the marker size range.

        Ensures that the lower bound of the range is less than the upper bound.
        If the bounds are in the wrong order, a ValueError is raised.

        Returns:
            Self: The instance itself, for method chaining.
        """
        # Get the lower and upper bounds of the marker size range
        low, high = self.Marker_size_range

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError("Limits for the Marker_size_range are in the wrong order")

        # Return the instance itself for method chaining
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
        low, high = self.Max_Eccentricity_range

        # Check if the lower bound is less than the upper bound
        if low >= high:
            # Raise a ValueError if the bounds are in the wrong order
            raise ValueError(
                "Limits for the Max_Eccentricity_range are in the wrong order"
            )

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
        low, high = self.Min_Solidity_range

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
