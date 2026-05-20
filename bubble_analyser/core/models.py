"""Core domain models for Bubble Analyser."""

import os
from pathlib import Path
from typing import cast
import numpy as np
from numpy import typing as npt

# Need to import MatLike if possible, or use Any
from typing import Any
from bubble_analyser.processing import calculate_px2mm


class InputFilesModel:
    """A model class for managing input image files and their paths.

    This class handles the selection, confirmation, and tracking of image files
    from a specified folder. It maintains lists of image paths in different formats
    for use by the UI and processing components.

    Attributes:
        sample_images_confirmed (bool): Flag indicating whether the folder selection
            has been confirmed.
        folder_path (Path): Path to the selected folder containing images.
        image_list (list[str]): List of image filenames without full paths.
        image_list_full_path (list[str]): List of full string paths to images for
            UI handlers.
        image_list_full_path_in_path (list[Path]): List of full Path objects for
            processing models.
        current_image_idx (int): Index of the currently selected image.
    """

    def __init__(self) -> None:
        """Initialize the InputFilesModel with default empty values."""
        self.sample_images_confirmed: bool = False
        self.folder_path: Path = cast(Path, None)

        self.image_list: list[str] = []

        # full path for ui event handlers
        self.image_list_full_path: list[str] = []

        # full path for processing models
        self.image_list_full_path_in_path: list[Path] = []
        self.current_image_idx: int = 0

    def confirm_folder_selection(self, folder_path: str) -> None:
        """Confirm the selected folder and populate image lists.

        This method sets the folder path, retrieves the list of images from the folder,
        and converts string paths to Path objects for the processing models.

        Args:
            folder_path (str): The path to the folder containing images.
        """
        self.folder_path = Path(folder_path)
        _ = self.get_image_list(folder_path)

        for path in self.image_list_full_path:
            self.image_list_full_path_in_path.append(Path(path))

        self.sample_images_confirmed = True

    def get_image_list(self, folder_path: str = cast(str, None)) -> tuple[list[str], list[str]]:
        """Get lists of image files from the specified folder.

        This method scans the specified folder for image files with supported extensions
        and populates both the filename list and full path list.

        Args:
            folder_path (str, optional): The path to scan for images. Defaults to None.

        Returns:
            tuple[list[str], list[str]]: A tuple containing:
                - List of image filenames
                - List of full paths to those images
        """
        self.image_list = []
        self.image_list_full_path = []

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self.image_list.append(file_name)
                self.image_list_full_path.append(os.path.join(folder_path, file_name))
        return self.image_list, self.image_list_full_path

    def reset(self) -> None:
        """Reset the model to its initial state."""
        self.image_list = []
        self.image_list_full_path = []
        self.image_list_full_path_in_path = []
        self.sample_images_confirmed = False


class CalibrationModel:
    """A model class for managing calibration data and pixel-to-millimeter conversion.

    This class handles the calibration process, including loading and processing
    calibration images, calculating the pixel-to-millimeter ratio, and managing
    Sbackground image correction.

    Attributes:
        pixel_img_confirmed (bool): Flag indicating whether the pixel calibration
            image has been confirmed.
        bknd_img_confirmed (bool): Flag indicating whether the background image
            has been confirmed.
        bknd_img_path (Path): Path to the background image file.
        bknd_img (npt.NDArray[np.int_]): Array containing the background image
            data.
        if_bknd (bool): Flag indicating whether a background image is being used.
        pixel_img_path (Path): Path to the pixel calibration image file.
        pixel_img (npt.NDArray[np.int_]): Array containing the pixel calibration
            image data.
        px2mm (float): The calculated pixel-to-millimeter conversion ratio.
        calibration_confirmed (bool): Flag indicating whether the calibration has
            been confirmed.
    """

    def __init__(self) -> None:
        """Initialize the CalibrationModel with default empty values."""
        self.pixel_img_confirmed: bool = False
        self.bknd_img_confirmed: bool = False

        self.bknd_img_path: Path = cast(Path, None)
        self.bknd_img: npt.NDArray[np.int_]
        self.if_bknd: bool = False

        self.pixel_img_path: Path
        self.pixel_img: npt.NDArray[np.int_]

        self.px2mm: float = 0.0
        self.px2mm_display: float = 0.0
        self.calibration_confirmed: bool = False

    def get_px2mm_ratio(
        self,
        pixel_img_path: Path,
        gui: object = None,
    ) -> tuple[float, Any]:
        """Calculate the pixel-to-millimeter ratio from a calibration image.

        This method uses the calculate_px2mm function to determine the conversion
        ratio between pixels and millimeters based on a calibration image
        (typically containing a ruler).

        Args:
            pixel_img_path (Path): Path to the calibration image file.
            gui (object, optional): GUI object for displaying interactive elements.
                Defaults to None.

        Returns:
            float: The calculated pixel-to-millimeter ratio.
            img_drawed_line: The ruler image with the drawn line.
        """
        __, self.px2mm, img_drawed_line = calculate_px2mm(pixel_img_path, gui)  # type: ignore
        return self.px2mm, img_drawed_line

    def confirm_calibration(self) -> None:
        """Mark the calibration as confirmed.

        This method sets the calibration_confirmed flag to True, indicating that
        the calibration process has been completed and confirmed by the user.
        """
        self.calibration_confirmed = True

    def reset(self) -> None:
        """Reset the model to its initial state."""
        self.calibration_confirmed = False
