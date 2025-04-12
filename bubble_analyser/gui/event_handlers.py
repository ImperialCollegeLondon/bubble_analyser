"""Event handlers module for the Bubble Analyser application.

This module provides a comprehensive set of event handlers and controllers for managing
user interactions within the GUI. It handles various aspects of the application's
functionality across different tabs and components.

Key components include:
- FolderTabHandler: Manages folder selection and image list management
- CalibrationTabHandler: Handles calibration processes and pixel-to-mm conversion
- ImageProcessingTabHandler: Controls image processing operations and parameter adjustments
- ExportSettingsHandler: Manages export settings and file saving operations
- TomlFileHandler: Handles configuration file loading and validation

Each handler is designed to manage specific aspects of the user interface while
maintaining separation of concerns and providing clear interaction patterns between
the GUI and the underlying processing logic.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import toml as tomllib  # type: ignore
from numpy import typing as npt
from pydantic import ValidationError
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QListView,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidgetItem,
    QTreeView,
    QVBoxLayout,
)

from bubble_analyser.gui import (
    CalibrationModel,
    ImageProcessingModel,
    InputFilesModel,
    WorkerThread,
)
from bubble_analyser.gui.gui import MainWindow as MainWindow
from bubble_analyser.processing import Config, cv2_to_qpixmap


class ExportSettingsHandler(QDialog):
    """A dialog for configuring export settings for processed images.

    This class provides a user interface for selecting and confirming the directory
    where processed images will be saved.

    Attributes:
        save_path (Path): The directory path where processed images will be saved.
        default_path_edit (QLineEdit): Text field displaying the current save path.
        confirm_button (QPushButton): Button to confirm the selected path.
    """

    def __init__(self, parent=None) -> None:  # type: ignore
        """Initialize the export settings dialog.

        Args:
            parent: The parent widget. Defaults to None.
            params (Config, optional): Configuration parameters containing the default save path. Defaults to None.
        """
        super().__init__(parent)

        self.setWindowTitle("Export Settings")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout(self)

        self.save_path: Path = cast(Path, "Select Export Path")
        self.if_save_path: bool = False

        # Default path for results saving
        self.default_path_edit = QLineEdit()
        self.default_path_edit.setPlaceholderText(str(self.save_path))  # type: ignore
        select_folder_button = QPushButton("Select Folder")
        select_folder_button.clicked.connect(self.select_folder)

        # Confirm folder selection
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.default_path_edit)
        path_layout.addWidget(select_folder_button)

        path_Vlayout = QVBoxLayout()
        path_Vlayout.addLayout(path_layout)
        path_Vlayout.addWidget(self.confirm_button)

        layout.addLayout(path_Vlayout)

    def accept(self) -> None:
        """Handle the confirmation of the selected save path.

        Validates the path before accepting the dialog. If the path is invalid,
        the dialog remains open.

        Returns:
            None
        """
        if self.check_if_path_valid() is False:
            return None
        super().accept()
        self.save_path = cast(Path, self.default_path_edit.text())
        logging.info(
            f"Export path for final graph, csv datafile, and processed\
images (optional) set as: {self.save_path}"
        )
        self.if_save_path = True

    def select_folder(self) -> None:
        """Open a file dialog to select a folder for saving processed images.

        Updates the text field with the selected folder path.
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", "", QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        if folder_path:
            self.default_path_edit.setText(folder_path)

    def check_if_path_valid(self) -> bool:
        """Check if the selected path is valid and accessible.

        Returns:
            bool: True if the path is valid, False otherwise.
        """
        try:
            Path(self.save_path).resolve()
            return True
        except FileNotFoundError as e:
            error_str = str(e)
            QMessageBox.warning(self, "Error", error_str)
            return False


class TomlFileHandler:
    """A handler class for loading and validating TOML configuration files.

    This class is responsible for reading configuration parameters from a TOML file,
    validating them against the Config schema, and providing access to the parsed parameters.

    Attributes:
        file_path (Path): Path to the TOML configuration file.
        params (Config): Validated configuration parameters loaded from the file.
        gui: Reference to the GUI instance for displaying warnings.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the TOML file handler with the specified file path.

        Args:
            file_path (Path): Path to the TOML configuration file to load.
        """
        self.file_path: Path = file_path
        self.params: Config
        self.load_toml()

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance for displaying warnings.

        This method should be called after the GUI has been initialized to enable
        the handler to interact with GUI components and display warning messages.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def load_toml(self) -> None:
        """Load and validate the TOML configuration file.

        Attempts to parse the TOML file and validate its contents against the Config schema.
        If validation fails, an error message is displayed.
        """
        try:
            self.params = Config(**tomllib.load(self.file_path))
        except ValidationError as e:
            error_str = str(e)
            print(error_str)
            self._show_warning("Error in Config File Setting", error_str)

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class FolderTabHandler:
    """Handler for the folder selection tab in the GUI.

    This class manages the selection and confirmation of folders containing image files
    for processing, and handles the preview of selected images.

    Attributes:
        model (InputFilesModel): Model for managing input image files and paths.
        image_path (Path): Default path for raw images from configuration.
        gui: Reference to the main GUI instance.
    """

    def __init__(self, model: InputFilesModel, params: Config) -> None:
        """Initialize the folder tab handler with model and configuration parameters.

        Args:
            model (InputFilesModel): Model for managing input image files.
            params (Config): Configuration parameters containing default paths.
        """
        self.model: InputFilesModel = model
        self.image_path: Path = params.raw_img_path

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def select_folder(self) -> None:
        """Handle the folder selection process.

        Opens a file dialog for the user to select a folder containing images.
        The dialog shows both folders and image files to help with selection.
        If a folder has already been confirmed, displays a warning instead.
        """
        if self.model.sample_images_confirmed:
            self._show_warning("Selection Locked", "You have already confirmed the folder selection.")
            return

        # Create a custom file dialog that shows both folders and image files
        file_dialog = QFileDialog(self.gui, "Select Folder Containing Images")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        file_dialog.setNameFilter(
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.TIF *.TIFF)"
        )

        # Show files as well as folders
        file_view = file_dialog.findChild(QListView, "listView")
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Also set this for the tree view
        tree_view = file_dialog.findChild(QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Initialize folder_path to None
        folder_path = None

        # Execute the dialog
        if file_dialog.exec():
            folder_path = file_dialog.selectedFiles()[0]

        if folder_path:
            self._update_folder_path(folder_path)
            self._populate_image_list(folder_path)
            logging.info(f"Raw image path set as: {folder_path}")
        else:
            logging.info("User cancelled folder selection")

    def _update_folder_path(self, folder_path: str) -> None:
        """Update the model and GUI with the selected folder path.

        Args:
            folder_path (str): The path to the selected folder.
        """
        self.model.folder_path = cast(Path, folder_path)
        self.gui.folder_path_edit.setText(folder_path)

    def _populate_image_list(self, folder_path: str) -> None:
        """Populate the image list widget with images from the selected folder.

        Args:
            folder_path (str): The path to the folder containing images.
        """
        images, _ = self.model.get_image_list(folder_path)
        self.gui.image_list.clear()
        self.gui.image_list.addItems(images)

    def confirm_folder_selection(self) -> None:
        """Confirm the selected folder and proceed to the next tab.

        Updates the model with the selected folder path, populates the image list,
        and switches to the calibration tab if a folder has been selected.
        """
        if self.model.sample_images_confirmed:
            self._show_warning("Selection Locked", "You have already confirmed the folder selection.")
            return

        folder_path = self.gui.folder_path_edit.text()
        if folder_path:
            self._update_folder_path(folder_path)
            self._populate_image_list(folder_path)
            self.model.confirm_folder_selection(folder_path)
            self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.calibration_tab))
            logging.info("Raw image path confirmed and locked.")
            logging.info("******************************Calibration session******************************")

    def preview_image_folder_tab(self) -> None:
        """Display a preview of the selected image in the folder tab.

        Loads the selected image from the list and displays it in the preview area,
        maintaining the aspect ratio.
        """
        selected_image = self.gui.image_list.currentItem().text()
        folder_path = self.gui.folder_path_edit.text()
        image_path = folder_path + "/" + selected_image
        pixmap = QPixmap(image_path)

        self.gui.image_preview.setPixmap(
            pixmap.scaled(
                self.gui.image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class CalibrationTabHandler:
    """Handler for the calibration tab in the GUI.

    This class manages the calibration process, including selection and processing of
    ruler images for pixel-to-millimeter conversion and optional background image correction.

    Attributes:
        calibration_model (CalibrationModel): Model for managing calibration data.
        img_resample (float): Resampling factor for image processing.
        px_img_path (Path): Default path for the ruler calibration image.
        gui: Reference to the main GUI instance.
    """

    def __init__(self, calibration_model: CalibrationModel, params: Config) -> None:
        """Initialize the calibration tab handler.

        Args:
            calibration_model (CalibrationModel): Model for managing calibration data.
            params (Config): Configuration parameters containing default values.
        """
        self.calibration_model: CalibrationModel = calibration_model
        self.img_resample: float = params.resample
        self.px_img_path: Path = params.ruler_img_path

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def select_ruler_button(self) -> None:
        """Handle the process of resolution calibration with ruler image selection and pixel-to-mm conversion."""
        # First select the image
        image_selected = self.select_px_mm_image()

        # Only proceed to calculate ratio if an image was actually selected
        if image_selected:
            self.get_px2mm_ratio()
        else:
            logging.info("Skipping px2mm ratio calculation as no image was selected")

    def select_px_mm_image(self) -> bool:
        """Handle the selection of a ruler image for pixel-to-millimeter calibration.

        Opens a file dialog for selecting a ruler image and updates the preview in the GUI.
        If calibration has already been confirmed, displays a warning instead.

        Returns:
            bool: True if an image was selected, False otherwise.
        """
        if self.calibration_model.calibration_confirmed:
            self._show_warning(
                "Selection Locked",
                "You have already confirmed the ruler image selection.",
            )
            return False

        # Use getOpenFileName with explicit options for better Windows compatibility
        image_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "Select Ruler Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.TIF *.TIFF)",
            options=QFileDialog.Option.ReadOnly,
        )

        if image_path:
            self.gui.pixel_img_name.setText(image_path)
            pixmap = QPixmap(image_path)
            self.gui.pixel_img_preview.setPixmap(
                pixmap.scaled(
                    self.gui.pixel_img_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            return True
        else:
            logging.info("User cancelled ruler image selection")
            return False

    def get_px2mm_ratio(self) -> None:
        """Calculate the pixel-to-millimeter ratio from the selected ruler image.

        If calibration has already been confirmed or no image is selected, displays
        appropriate warning messages.
        """
        if self.calibration_model.calibration_confirmed:
            self._show_warning("Selection Locked", "You have already confirmed the pixel-to-mm ratio.")
            return

        img_path: Path = cast(Path, self.gui.pixel_img_name.text())
        if os.path.exists(img_path):
            px2mm, img_drawed_line = self.calibration_model.get_px2mm_ratio(
                pixel_img_path=img_path, img_resample=self.img_resample, gui=self.gui
            )
            self.gui.manual_px_mm_input.setText(f"{px2mm:.3f}")

            pixmap = cv2_to_qpixmap(img_drawed_line)
            self.gui.pixel_img_preview.setPixmap(
                pixmap.scaled(
                    self.gui.pixel_img_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            logging.info(f"Pixel to millimeter ratio detected as: {px2mm:.3f}")
        else:
            self.gui.statusBar().showMessage("Image file does not exist or not selected.", 5000)

    def select_bg_corr_image(self) -> None:
        """Handle the selection of a background correction image.

        Opens a file dialog for selecting a background image and updates the preview in the GUI.
        If calibration has already been confirmed, displays a warning instead.
        """
        if self.calibration_model.calibration_confirmed:
            self._show_warning(
                "Selection Locked",
                "You have already confirmed the background image selection.",
            )
            return

        # Use getOpenFileName with explicit options for better Windows compatibility
        image_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "Select Background Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.TIF *.TIFF)",
            options=QFileDialog.Option.ReadOnly,
        )

        if image_path:
            self.calibration_model.bknd_img_path = Path(image_path)
            self.gui.bg_corr_image_name.setText(image_path)
            pixmap = QPixmap(image_path)
            self.gui.bg_corr_image_preview.setPixmap(
                pixmap.scaled(
                    self.gui.bg_corr_image_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            self.calibration_model.if_bknd = True
            logging.info(f"Background image path set as: {image_path}")
        else:
            logging.info("User cancelled background image selection")

    def confirm_calibration(self) -> None:
        """Confirm the calibration settings and proceed to the next tab.

        Updates the calibration model with the final values and switches to the
        image processing tab. If calibration has already been confirmed, displays
        a warning instead. The first image of the list will be previewed in the
        image processing tab.
        """
        if self.calibration_model.calibration_confirmed:
            self.gui.manual_px_mm_input.setText(f"{self.calibration_model.px2mm:.3f}")
            self._show_warning("Selection Locked", "You have already confirmed the calibration.")
            return

        self.calibration_model.bknd_img_path = Path(self.gui.bg_corr_image_name.text())
        self.calibration_model.px2mm = float(self.gui.manual_px_mm_input.text())
        self.calibration_model.confirm_calibration()

        self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.image_processing_tab))
        self.preview_image_intialize()
        logging.info(f"Pixel to millimeter ratio locked as: {self.calibration_model.px2mm:.3f}")
        logging.info(f"Background image path locked as: {self.calibration_model.bknd_img_path}")
        logging.info("Calibration confirmed and locked.")
        logging.info("******************************Parameter Adjustment Session******************************")

    def preview_image_intialize(self) -> None:
        """Display a preview of the first image in the image processing tab.

        Loads the selected image from the list and displays it in the preview area,
        maintaining the aspect ratio.
        """
        self.gui.image_list.setCurrentRow(0)
        selected_image = self.gui.image_list.currentItem().text()
        folder_path = self.gui.folder_path_edit.text()
        image_path = folder_path + "/" + selected_image
        pixmap = QPixmap(image_path)

        self.gui.sample_image_preview.setPixmap(
            pixmap.scaled(
                self.gui.sample_image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class ImageProcessingTabHandler(QThread):
    """Handler for the image processing tab in the GUI.

    This class manages the image processing operations, including algorithm selection,
    parameter configuration, and batch processing. It extends QThread to handle
    long-running processing tasks without blocking the GUI.

    Attributes:
        batch_processing_done (Signal): Signal emitted when batch processing completes.
        model (ImageProcessingModel): Model for managing image processing operations.
        params (Config): Configuration parameters for processing.
        current_index (int): Index of the currently selected image.
        algorithm_list (list[str]): List of available processing algorithms.
        export_handler (ExportSettingsHandler): Handler for export settings.
        if_save_processed_images (bool): Flag indicating whether to save processed images.
        temp_param_dict (dict[str, int | float]): Temporary storage for processing parameters.
        temp_filter_param_dict (dict[str, float]): Temporary storage for filter parameters.
        save_path (Path): Path where processed images will be saved.
    """

    batch_processing_done = Signal()
    check_for_export_path = Signal()

    def __init__(self, image_processing_model: ImageProcessingModel, params: Config) -> None:
        """Initialize the image processing tab handler.

        Args:
            image_processing_model (ImageProcessingModel): Model for managing image processing.
            params (Config): Configuration parameters for processing.
        """
        super().__init__()
        self.model: ImageProcessingModel = image_processing_model
        self.params: Config = params
        self.params_checker: Config = self.params.model_copy()
        self.current_index: int = 0
        self.algorithm_list: list[str] = []
        self.export_handler: ExportSettingsHandler
        self.if_save_processed_images = False

        self.save_path: Path = cast(Path, None)

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def pass_filter_params(self) -> None:
        """Pass filter parameters to the processing model.

        Args:
            filter_param_dict (dict[str, int | float]): Dictionary of filter parameters.
        """
        self.model.load_filter_params(self.filter_param_dict_1, self.filter_param_dict_2)

    def preview_image(self) -> None:
        """Display a preview of the currently selected image.

        Loads the selected image from the list and displays it in the preview area,
        maintaining the aspect ratio.
        """
        selected_image = self.gui.image_list.currentItem().text()
        folder_path = self.gui.folder_path_edit.text()
        image_path = folder_path + "/" + selected_image
        pixmap = QPixmap(image_path)

        self.gui.sample_image_preview.setPixmap(
            pixmap.scaled(
                self.gui.sample_image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )

    def check_params(self, name: str, value: int | float | str) -> bool:
        """Validate a parameter value against the configuration schema.

        A new Config instance is being created so the validation error can be refreshed every time.

        Args:
            name (str): The name of the parameter to validate.
            value (int | float): The value to validate.

        Returns:
            bool: True if the parameter is valid, False otherwise.
        """
        new_checker: Config = self.params_checker.model_copy()

        logging.info(f"Checking parameter: {name} {value}")
        if name == "element_size":
            try:
                new_checker.element_size = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid Element Size", str(e))
                return False

        if name == "connectivity":
            try:
                new_checker.connectivity = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid Connectivity", str(e))
                return False

        if name == "resample":
            try:
                new_checker.resample = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Resample Factor", str(e))
                return False

        if name == "max_thresh":
            try:
                new_checker.max_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "min_thresh":
            try:
                new_checker.min_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "step_size":
            try:
                new_checker.step_size = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Step Size", str(e))
                return False

        if name == "high_thresh":
            try:
                new_checker.high_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "mid_thresh":
            try:
                new_checker.mid_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "low_thresh":
            try:
                new_checker.low_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Thresh value", str(e))
                return False

        if name == "max_eccentricity":
            try:
                new_checker.max_eccentricity = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Max Eccentricity", str(e))
                return False

        if name == "min_solidity":
            try:
                new_checker.min_solidity = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Solidity", str(e))
                return False

        if name == "min_size":
            try:
                new_checker.min_size = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Size", str(e))
                return False

        if name == "L_maxA":
            try:
                new_checker.L_maxA = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Max Area for Large bubbles", str(e))
                return False

        if name == "L_minA":
            try:
                new_checker.L_minA = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Area for Large bubbles", str(e))
                return False

        if name == "s_maxA":
            try:
                new_checker.s_maxA = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Max Area for Small bubbles", str(e))
                return False

        if name == "s_minA":
            try:
                new_checker.s_minA = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Area for Small bubbles", str(e))
                return False
        return True

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)

    def update_sample_image(self, direction: str) -> None:
        """Navigate between images in the image list.

        Args:
            direction (str): The direction to navigate, either "prev" or "next".
        """
        current_row = self.gui.image_list.currentRow()

        if direction == "prev":
            if current_row > 0:
                current_row -= 1
                self.gui.image_list.setCurrentRow(current_row)
        elif direction == "next":
            if current_row < self.gui.image_list.count() - 1:
                current_row += 1
                self.gui.image_list.setCurrentRow(current_row)

        self.current_index = current_row
        logging.info(f"(event_handlers/update_sample_img)Current image index: {self.current_index}")

        self.preview_image()
        self.update_preview_procsd_img_button()

    def update_preview_procsd_img_button(self) -> None:
        """Check if the current image are being processed with segmentation and filtering.

        If the current image has been processed, the preview processed image button is enabled.
        """
        if_img, img_before_filter, img_after_filter = self.model.preview_processed_image(self.current_index)
        if if_img:
            logging.info("Preview processed image enabled for current image.")
            self.gui.preview_processed_images_button.setEnabled(True)

        else:
            self.gui.preview_processed_images_button.setEnabled(False)

    def preview_processed_images(self) -> None:
        """Preview the processed images for the current image.

        Retrieves the processed images from the model and displays them in the preview areas.
        If the image hasn't been processed yet, displays a warning.
        """
        if_img, img_before_filter, img_after_filter = self.model.preview_processed_image(self.current_index)

        if if_img:
            self.update_label_before_filtering(img_before_filter)
            self.update_process_image_preview(img_after_filter)
        else:
            self._show_warning("Image Not Found", "Image has not been fully processed yet.")

    # -------Second Column Functions-------------------------
    def initialize_algorithm_combo(self) -> None:
        """Initialize the algorithm combo box with available processing methods.

        Populates the combo box with the names of all available processing algorithms
        and sets the default algorithm.
        """
        # Initialize the algorithm combo box
        # And achieve all the available methods' names

        logging.info("Initializing algorithm combo box...")
        for algorithm, params in self.model.all_methods_n_params.items():
            logging.info(f"Initialize algorithm: {algorithm}")
            self.algorithm_list.append(algorithm)

        self.gui.algorithm_combo.addItems(self.algorithm_list)
        self.update_model_algorithm(self.algorithm_list[0])
        logging.info("Algorithm combo box initialized.")

    def load_parameter_table_1(self, algorithm: str) -> None:
        """Load the parameter table with values for the selected algorithm.

        Args:
            algorithm (str): The name of the algorithm whose parameters should be loaded.
        """
        # pass the current algorithm text in the gui to the model
        # This function only triggered by first initialization and algorithm change

        self.current_algorithm = algorithm
        logging.info(f"Current choosing algorithm: {self.current_algorithm}")

        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.current_algorithm:
                self.gui.param_sandbox1.setRowCount(len(params))

                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    self.gui.param_sandbox1.setItem(row, 0, QTableWidgetItem(name))
                    self.gui.param_sandbox1.setItem(row, 1, QTableWidgetItem(str(value)))

                break

    def handle_algorithm_change(self, new_algorithm: str) -> None:
        """Handle a change in the selected algorithm.

        Updates the parameters in the model, reloads the parameter table,
        and updates the model with the new algorithm.

        Args:
            new_algorithm (str): The name of the newly selected algorithm.
        """
        # Update the algorithm in the model
        self.update_segment_parameters()

        # Reload the param table in the GUI
        self.load_parameter_table_1(new_algorithm)

        # Update params in the model
        self.update_model_algorithm(new_algorithm)

    def update_model_algorithm(self, algorithm: str) -> None:
        """Update the algorithm in the processing model.

        Args:
            algorithm (str): The name of the algorithm to set in the model.
        """
        self.model.algorithm = algorithm

    def confirm_parameter_before_filtering(self) -> None:
        """Confirm the parameters for the first step of image processing.

        Validates all parameters against the configuration schema before
        proceeding with the first processing step.
        """
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("-------------------------------Running Step 1: Segmentation------------------------------")
        logging.info("-----------------------------------------------------------------------------------------")
        self.update_segment_parameters()

        # Validate the parameters
        logging.info("------------------------------Validating Segment Parameters------------------------------")
        # Update the model parameters
        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    if_valid = self.check_params(name, value)
                    if not if_valid:
                        return

        # self.pass_segment_params(self.model.segment_param_dict)
        self._process_step_1()

    def convert_value(self, text: str) -> int | float | str:
        """Convert a string to the appropriate numeric type.

        Args:
            text (str): The string to convert.

        Returns:
            int | float | str: The converted value, as an integer if the value is an integer,
                               as a float if the value is a floating-point number,
                               or as a string if the value cannot be converted to a number.
        """
        try:
            value = float(text)
            # Return an int if the number is integer
            if value.is_integer():
                logging.info(f"{value} determined as integer.")
                return int(value)
            else:
                logging.info(f"{value} determined as float.")
                return value
            # return int(value) if value.is_integer() else value
        except ValueError:
            # Fallback if not a number
            return text

    def extract_parameters_from_table(self, table_widget) -> dict[str, int | float]:  # type: ignore
        """Extract parameters from a table widget.

        Args:
            table_widget: The table widget containing parameter names and values.

        Returns:
            dict[str, int|float]: A dictionary mapping parameter names to their values.
        """
        params = {}
        row_count = table_widget.rowCount()
        for row in range(row_count):
            name_item = table_widget.item(row, 0)
            value_item = table_widget.item(row, 1)
            if name_item and value_item:
                params[name_item.text()] = self.convert_value(value_item.text())
        return params  # type: ignore

    def update_segment_parameters(self) -> bool:
        """Update the segment parameters in the model from the GUI table.

        Extracts parameters from the table widget and updates the corresponding
        parameters in the model for the currently selected algorithm.

        Returns:
            bool: True if the parameters were successfully updated.
        """
        # Update the params in the model
        # Extract parameters from the table
        logging.info("------------------------------Updating Parameters------------------------------")
        params = self.extract_parameters_from_table(self.gui.param_sandbox1)
        # Update the model's dictionary for the selected algorithm
        for algorithm_name, params_in_dict in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for name, value in params.items():
                    logging.info(f"Updating {name} to {value}")
                    params_in_dict[name] = value

        return True

    def _process_step_1(self) -> None:
        """Execute the first step of image processing.

        Calls the model's step_1_main method to process the current image
        and updates the preview with the results.
        """
        logging.info("------------------------------Processing Started------------------------------")
        step_1_img = self.model.step_1_main(self.current_index)
        self.update_label_before_filtering(step_1_img)

    def update_label_before_filtering(self, img: npt.NDArray[np.int_]) -> None:
        """Update the preview of the image after the first processing step.

        Args:
            img (npt.NDArray[np.int_]): The processed image to display.
        """
        self.gui.label_before_filtering.axes.clear()
        self.gui.label_before_filtering.axes.imshow(img)
        self.gui.label_before_filtering.draw()

    # -------Third Column Functions: Filtering-------------------------
    def initialize_parameter_table_2(self) -> None:
        """Initialize the filtering parameters table with default values.

        Populates the filtering parameters table with the current values from
        the temporary filter parameter dictionary.
        """
        logging.info("Initializing parameter table 2...")
        self.filter_param_dict_1 = self.model.filter_param_dict_1
        self.filter_param_dict_2 = self.model.filter_param_dict_2

        self.gui.param_sandbox2.setRowCount(len(self.filter_param_dict_1))
        row = 0
        for property, value in self.filter_param_dict_1.items():
            logging.info(f"Filter param name: {property}, value: {value}")
            self.gui.param_sandbox2.setItem(row, 0, QTableWidgetItem(property))
            self.gui.param_sandbox2.setItem(row, 1, QTableWidgetItem(str(value)))
            row += 1

        logging.info("Parameter table 2 initialized.")

    def handle_find_circles(self) -> None:
        """Toggle the visibility of the circle parameter box based on checkbox state.

        This method responds to the state of the find circles checkbox (fc_checkbox):
        - When checked: Shows the circle parameter box, sets find_circles(Y/N) to "Y",
          and populates the parameter table with values from filter_param_dict_2
        - When unchecked: Hides the circle parameter box and sets find_circles(Y/N) to "N"

        The circle parameter box displays all parameters from filter_param_dict_2 except
        for the find_circles(Y/N) parameter itself. Each parameter is displayed in a row
        with its name in the first column and its value in the second column.
        """
        state = self.gui.fc_checkbox.isChecked()
        if state:
            logging.info("Find circles enabled.")
            self.gui.circle_param_box.show()
            self.filter_param_dict_2["find_circles(Y/N)"] = "Y"
            self.gui.circle_param_box.setRowCount(len(self.filter_param_dict_2) - 1)
            logging.info("Find circles parameter box set as Visible.")

            row = 0
            for property, value in self.filter_param_dict_2.items():
                if property != "find_circles(Y/N)":
                    logging.info(f"Filter parameter name: {property}, value: {value}")
                    self.gui.circle_param_box.setItem(row, 0, QTableWidgetItem(property))
                    self.gui.circle_param_box.setItem(row, 1, QTableWidgetItem(str(value)))
                    row += 1
        else:
            logging.info("Find circles disabled.")
            self.gui.circle_param_box.hide()
            self.filter_param_dict_2["find_circles(Y/N)"] = "N"

    def confirm_parameter_for_filtering(self) -> None:
        """Confirm the filtering parameters and apply them to the current image.

        Validates all filtering parameters against the configuration schema before
        proceeding with the second processing step.
        """
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("--------------------------------Running Step 2: Filtering--------------------------------")
        logging.info("-----------------------------------------------------------------------------------------")

        logging.info("------------------------------Updating Filtering Parameters------------------------------")
        self.store_filter_params()

        logging.info("------------------------------Validating Filter Parameters------------------------------")
        for name, value in self.filter_param_dict_1.items():
            if_valid = self.check_params(name, value)
            if not if_valid:
                return

        for name, value in self.filter_param_dict_2.items():
            if_valid = self.check_params(name, value)
            if not if_valid:
                return

        self.pass_filter_params()
        self._process_step_2()

    def store_filter_params(self) -> None:
        """Store the filtering parameters from the GUI table into the filter parameter dictionary.

        Extracts the circle parameters from the circle parameter table widget and updates the
        filter parameter dictionary with the new values.
        """
        logging.info("Storing filter parameters...")
        if not hasattr(self.gui, "circle_param_box"):
            pass
        else:
            for row in range(self.gui.circle_param_box.rowCount()):
                name_item = self.gui.circle_param_box.item(row, 0)
                value_item = self.gui.circle_param_box.item(row, 1)
                if name_item and value_item:
                    param_name = name_item.text()
                    param_value = value_item.text()
                    self.filter_param_dict_2[param_name] = float(param_value)
                    logging.info(f"Updating {param_name} to {param_value}")

        for row in range(self.gui.param_sandbox2.rowCount()):
            name_item = self.gui.param_sandbox2.item(row, 0)
            value_item = self.gui.param_sandbox2.item(row, 1)
            if name_item and value_item:
                param_name = name_item.text()
                param_value = value_item.text()

                # Handle numeric parameter
                float_value = cast(float, param_value)
                self.filter_param_dict_1[param_name] = float_value

                logging.info(f"Updating {param_name} to {param_value}")

    def _process_step_2(self) -> None:
        """Execute the second step of image processing (filtering).

        Calls the model's step_2_main method to apply filtering to the current image
        and updates the preview with the results.
        """
        step_2_img = self.model.step_2_main(self.current_index)
        self.update_process_image_preview(step_2_img)

    # -------Third Column Functions: Manual Ellipse Adjustment-------------------------
    def ellipse_manual_adjustment(self) -> None:
        """Launch the ellipse adjustment tool for manual fine-tuning of detected ellipses.

        Opens a separate window with tools for manually adjusting the detected ellipses,
        and updates the preview with the adjusted ellipses when complete.
        """
        img = self.model.ellipse_manual_adjustment(self.current_index)
        self.update_process_image_preview(img)

    def update_process_image_preview(self, img: npt.NDArray[np.int_]) -> None:
        """Update the preview of the processed image after filtering or adjustment.

        Args:
            img (npt.NDArray[np.int_]): The processed image to display.
        """
        self.gui.processed_image_preview.axes.clear()
        self.gui.processed_image_preview.axes.imshow(img)
        self.gui.processed_image_preview.draw()

    # -------Third Column Functions: Batch Processing----------------------------
    def ask_if_batch(self) -> None:
        """Function to handle the batch processing of all images in the folder.

        Creates and displays a confirmation dialog with options for saving processed images.
        If confirmed, initiates the batch processing operation.
        """
        self.if_save_processed_images = False
        confirm_dialog = self.create_confirm_dialog()
        self.create_save_images_checkbox(confirm_dialog)

        response = confirm_dialog.exec()

        if response == QMessageBox.StandardButton.Ok:
            self.check_for_export_path.emit()
            # self.batch_process_images()
        else:
            logging.info("Batch processing canceled.")

    def create_confirm_dialog(self) -> QMessageBox:
        """Create a confirmation dialog for batch processing.

        Returns:
            QMessageBox: The configured confirmation dialog.
        """
        confirm_dialog = QMessageBox(self.gui)
        confirm_dialog.setWindowTitle("Batch Processing Confirmation")
        confirm_dialog.setText("The parameters will be applied to all the images. Confirm to process.")
        confirm_dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        return confirm_dialog

    def create_save_images_checkbox(self, dialog: QMessageBox) -> QCheckBox:
        """Create a checkbox for enabling image saving during batch processing.

        Args:
            dialog (QMessageBox): The dialog to add the checkbox to.

        Returns:
            QCheckBox: The configured checkbox widget.
        """
        save_images_checkbox = QCheckBox("Save processed images")
        save_images_checkbox.stateChanged.connect(
            lambda: self.update_if_save_processed_images(save_images_checkbox.isChecked())
        )
        dialog.setCheckBox(save_images_checkbox)
        return save_images_checkbox

    def update_if_save_processed_images(self, state: bool) -> None:
        """Update the flag indicating whether to save processed images.

        Args:
            state (bool): The new state of the save images flag.
        """
        self.if_save_processed_images = state

    def batch_process_images(self) -> None:
        """Execute batch processing on all images in the folder.

        Validates all parameters, initializes the progress window, and starts
        the worker thread to process all images. If saving is enabled, verifies
        that the save path exists before proceeding.
        """
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("--------------------------------Running Batch Processing--------------------------------")
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("------------------------------Validating Segment Parameters------------------------------")
        # Update the model parameters
        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    logging.info(f"Checking segment params before batch processing {name} {value}")
                    if_valid = self.check_params(name, value)
                    if not if_valid:
                        return

        logging.info("------------------------------Validating Filter Parameters------------------------------")
        self.store_filter_params()
        self.update_segment_parameters()
        self.pass_filter_params()
        self.show_progress_window(len(self.model.img_path_list))

        if self.if_save_processed_images:
            # check if save path exists
            if not os.path.exists(self.export_handler.save_path):
                QMessageBox.warning(
                    self.gui,
                    "Warning",
                    f"The save path {self.export_handler.save_path} does not exist. Please select a valid directory.",
                    QMessageBox.StandardButton.Ok,
                )
                return

        self.worker_thread = WorkerThread(self.model, self.if_save_processed_images, self.export_handler.save_path)
        self.worker_thread.update_progress.connect(self.update_progress_bar)
        self.worker_thread.processing_done.connect(self.on_processing_done)
        self.worker_thread.start()

    def show_progress_window(self, num_images: int) -> None:
        """Create and show a progress window with a loading bar.

        Args:
            num_images (int): The total number of images to process, used to set the progress bar range.
        """
        self.progress_dialog = QDialog(self.gui)
        self.progress_dialog.setWindowTitle("Batch Processing in Progress")
        self.progress_dialog.setFixedSize(400, 100)

        layout = QVBoxLayout(self.progress_dialog)

        self.progress_bar = QProgressBar(self.progress_dialog)
        self.progress_bar.setRange(0, num_images)
        self.progress_bar.setValue(0)  # Start with 0 progress

        layout.addWidget(self.progress_bar)

        self.progress_dialog.setLayout(layout)
        self.progress_dialog.show()

    def update_progress_bar(self, value: int) -> None:
        """Update the progress bar value.

        Args:
            value (int): The new progress value to display.
        """
        self.progress_bar.setValue(value)
        logging.info(f"Updating progress bar: {value}")

    def on_processing_done(self) -> None:
        """Handle the completion of image processing.

        Closes the progress dialog, emits the batch_processing_done signal,
        and switches to the results tab.
        """
        # Close the progress dialog
        self.progress_dialog.close()
        self.batch_processing_done.emit()

        # Switch to the final tab
        self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.results_tab))
        logging.info("Batch processing completed.")
        logging.info("******************************Result Session******************************")


class ResultsTabHandler(QThread):
    """Handler for the results tab in the GUI.

    This class manages the display and export of processing results, including
    histogram generation and descriptive statistics calculation.

    Attributes:
        params (Config): Configuration parameters for results display.
        save_path (Path): Path where results will be saved.
        ellipses_properties (list[list[dict[str, float]]]): Properties of detected ellipses for all images.
        export_handler (ExportSettingsHandler): Handler for export settings.
        gui: Reference to the main GUI instance.
    """

    check_for_export_path = Signal()

    def __init__(self, params: Config) -> None:
        """Initialize the results tab handler.

        Args:
            params (Config): Configuration parameters containing default values.
        """
        super().__init__()
        self.params = params
        self.save_path = params.save_path

        self.ellipses_properties: list[list[dict[str, float]]]
        self.export_handler: ExportSettingsHandler
        self.if_dinf_displayed: bool = False

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui
        # results tab
        self.gui.histogram_by.currentIndexChanged.connect(self.generate_histogram)
        self.gui.pdf_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.cdf_checkbox.stateChanged.connect(self.generate_histogram)

        self.gui.bins_spinbox.valueChanged.connect(self.generate_histogram)

        self.gui.min_x_axis_input.textChanged.connect(self.generate_histogram)
        self.gui.max_x_axis_input.textChanged.connect(self.generate_histogram)

        self.gui.legend_position_combobox.currentIndexChanged.connect(self.generate_histogram)
        self.gui.d32_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.dmean_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.dxy_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.dxy_x_input.textChanged.connect(self.generate_histogram)
        self.gui.dxy_y_input.textChanged.connect(self.generate_histogram)
        self.gui.save_button.clicked.connect(self.check_for_export_path.emit)

    def load_ellipse_properties(
        self,
        properties: list[list[dict[str, float]]],
        algorithm: str,
        all_methods_n_params: dict[str, dict[str, float | int]],
        param_dict_1: dict[str, float | str],
        param_dict_2: dict[str, float | str],
    ) -> None:
        """Load the properties of detected ellipses for display and analysis.

        Args:
            properties (list[list[dict[str, float]]]): Properties of detected ellipses for all images.
            algorithm (str): The name of the algorithm used for detection.
            all_methods_n_params (dict[str, dict[str, float | int]]): Dictionary containing
                parameters for all algorithms.
            param_dict_1 (dict[str, float | str]): Dictionary containing parameters for the
                first algorithm.
            param_dict_2 (dict[str, float | str]): Dictionary containing parameters for the
                second algorithm.
        """
        self.ellipses_properties = properties
        self.algorithm = algorithm
        self.all_methods_n_params = all_methods_n_params
        self.param_dict_1 = param_dict_1
        self.param_dict_2 = param_dict_2
        pass

    def generate_histogram(self) -> None:
        """Generate and display a histogram of bubble sizes.

        Creates a histogram showing the distribution of equivalent diameters of detected bubbles.
        Optionally displays PDF, CDF, and characteristic diameters (d32, dmean, dxy) based on
        user selections. Updates the plot with appropriate labels and legend.

        Supports two histogram types:
        - Count: Shows the count of bubbles for each diameter range
        - Volume: Shows the volume of bubbles (using diameter^3) for each diameter range
        """
        num_bins = self.gui.bins_spinbox.value()
        show_pdf = self.gui.pdf_checkbox.isChecked()
        show_cdf = self.gui.cdf_checkbox.isChecked()
        show_d32 = self.gui.d32_checkbox.isChecked()
        show_dmean = self.gui.dmean_checkbox.isChecked()
        show_dxy = self.gui.dxy_checkbox.isChecked()
        histogram_type = self.gui.histogram_by.currentText()

        logging.info(f"Histogram type: {histogram_type}")

        try:
            equivalent_diameters_array = self.get_equivalent_diameters_list()
        except AttributeError as e:
            error = str(e)
            self._show_warning(
                "Error in Histogram Generation",
                f"{error}, please process the images first.",
            )
            return

        x_min = float(np.min(equivalent_diameters_array))
        x_max = float(np.max(equivalent_diameters_array))

        # Clear current graph
        self.gui.histogram_canvas.axes.set_xlabel("")
        self.gui.histogram_canvas.axes.set_ylabel("")
        self.gui.histogram_canvas.axes.clear()

        try:
            if self.gui.histogram_canvas.axes2:
                self.gui.histogram_canvas.axes2.clear()
                self.gui.histogram_canvas.axes2.set_ylabel("")
                self.gui.histogram_canvas.axes2.set_yticklabels([])
                self.gui.histogram_canvas.axes2.set_yticks([])
                del self.gui.histogram_canvas.axes2

        except AttributeError:
            pass

        # Plot histogram based on selected type
        if histogram_type == "Volume":
            # Calculate volumes (diameter^3) for each bubble
            volumes = equivalent_diameters_array**3
            counts, bins, patches = self.gui.histogram_canvas.axes.hist(
                equivalent_diameters_array,
                bins=num_bins,
                range=(x_min, x_max),
                weights=volumes,  # Use volumes as weights
            )
            # Set graph labels for volume histogram
            self.gui.histogram_canvas.axes.set_xlabel("Equivalent diameter [mm]")
            self.gui.histogram_canvas.axes.set_ylabel("Volume [mm³]")
        else:  # Count histogram (default)
            counts, bins, patches = self.gui.histogram_canvas.axes.hist(
                equivalent_diameters_array, bins=num_bins, range=(x_min, x_max)
            )
            # Set graph labels for count histogram
            self.gui.histogram_canvas.axes.set_xlabel("Equivalent diameter [mm]")
            self.gui.histogram_canvas.axes.set_ylabel("Count [#]")

        # Calculate descriptive sizes
        d32, d_mean, dxy = self.calculate_descriptive_sizes(equivalent_diameters_array)
        if not self.if_dinf_displayed:
            logging.info(f"d32: {d32}, d_mean: {d_mean}, dxy: {dxy}")
            self.if_dinf_displayed = True

        # Update descriptive size label
        desc_text = f"Results:\nd32 = {d32:.2f} mm\ndmean = {d_mean:.2f} mm\ndxy = {dxy:.2f} mm"
        self.gui.descriptive_size_label.setText(desc_text)

        # Optionally add CDF and PDF
        if show_pdf or show_cdf:
            self.gui.histogram_canvas.axes2 = self.gui.histogram_canvas.axes.twinx()
            self.gui.histogram_canvas.axes2.set_ylabel("Probability [%]")

            # For volume histogram, we need to normalize differently
            if histogram_type == "Volume":
                total = np.sum(counts)
                if show_cdf:
                    cdf = np.cumsum(counts) / total * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], cdf, "r-", marker="o", label="CDF")
                if show_pdf:
                    pdf = counts / total * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], pdf, "b-", marker="o", label="PDF")
            else:  # Count histogram
                if show_cdf:
                    cdf = np.cumsum(counts) / np.sum(counts) * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], cdf, "r-", marker="o", label="CDF")
                if show_pdf:
                    pdf = counts / np.sum(counts) * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], pdf, "b-", marker="o", label="PDF")

        # Add characteristic diameter lines
        if show_d32:
            self.gui.histogram_canvas.axes.axvline(x=d32, color="r", linestyle="-", label="d32")

        if show_dmean:
            self.gui.histogram_canvas.axes.axvline(x=d_mean, color="g", linestyle="--", label="dmean")

        if show_dxy:
            self.gui.histogram_canvas.axes.axvline(x=dxy, color="b", linestyle="--", label="dxy")

        # Apply Legend Options
        legend_position = self.gui.legend_position_combobox.currentText()

        legend_location_map = {
            "North East": "upper right",
            "North West": "upper left",
            "South East": "lower right",
            "South West": "lower left",
        }

        logging.info(f"Legend_position: {legend_position}")

        # Add legend to the graph
        if show_cdf or show_pdf or show_d32 or show_dmean or show_dxy:
            lines1, labels1 = self.gui.histogram_canvas.axes.get_legend_handles_labels()
            if show_cdf or show_pdf:
                lines2, labels2 = self.gui.histogram_canvas.axes2.get_legend_handles_labels()
                self.gui.histogram_canvas.axes.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc=legend_location_map.get(legend_position, "upper right"),
                )

        # Redraw the canvas
        self.gui.histogram_canvas.draw()
        return

    def calculate_descriptive_sizes(self, equivalent_diameters: npt.NDArray[np.float64]) -> tuple[float, float, float]:
        """Calculate characteristic diameters from the equivalent diameters.

        Args:
            equivalent_diameters (npt.NDArray[np.float64]): Array of equivalent diameters.

        Returns:
            tuple[float, float, float]: A tuple containing (d32, d_mean, dxy) where:
                - d32: Sauter mean diameter
                - d_mean: Arithmetic mean diameter
                - dxy: General mean diameter with user-specified powers
        """
        dxy_x_power: int = int(self.gui.dxy_x_input.text())
        dxy_y_power: int = int(self.gui.dxy_y_input.text())
        d32: float = np.sum(equivalent_diameters**3) / np.sum(equivalent_diameters**2)  # type: ignore

        # d32, Sauter diameter, should be calculated based on the area, and volume
        # diameter of a circle, which is unkown right now

        d_mean: float = float(np.mean(equivalent_diameters))
        dxy: float = np.sum(equivalent_diameters**dxy_x_power) / np.sum(equivalent_diameters**dxy_y_power)  # type: ignore

        return d32, d_mean, dxy

    def get_equivalent_diameters_list(self) -> npt.NDArray[np.float64]:
        """Extract equivalent diameters from all detected ellipses.

        Returns:
            npt.NDArray[np.float64]: Array of equivalent diameters from all processed images.
        """
        equivalent_diameters = []
        for image in self.ellipses_properties:
            for ellipse in image:
                equivalent_diameters.append(ellipse["equivalent_diameter"])

        equivalent_diameters_array = np.array(equivalent_diameters)
        return equivalent_diameters_array

    def save_results(self) -> None:
        """Saves histogram and data to the selected folder."""
        folder_path = self.export_handler.save_path
        if folder_path == "" or not os.path.exists(folder_path):
            self._show_warning("Folder Not Found", "Please select a valid folder in export settings.")
            return

        # Get the user-specified filenames
        graph_filename = self.gui.graph_filename_edit.text()
        csv_filename = self.gui.csv_filename_edit.text()

        if not graph_filename or not csv_filename:
            QMessageBox.warning(
                self.gui,
                "Filename Missing",
                "Please provide filenames for both graph and CSV files.",
            )
            return

        # Set file paths
        graph_path = os.path.join(folder_path, f"{graph_filename}.png")
        csv_path = os.path.join(folder_path, f"{csv_filename}.csv")
        config_path = os.path.join(folder_path, f"{csv_filename}_config.csv")

        # Assuming `self.histogram_canvas` is a matplotlib canvas
        self.save_graph(graph_path)
        self.save_ellipses_data(csv_path)
        self.save_config_data(config_path)

        self._show_warning("Results Saved", f"Results have been saved successfully to {folder_path}.")
        logging.info(f"Results have been saved successfully to {folder_path}.")
        return

    def save_graph(self, export_path: str) -> None:
        """Saves the current histogram to the selected folder."""
        self.gui.histogram_canvas.fig.savefig(export_path)

    def save_ellipses_data(self, export_path: str) -> None:
        """Saves the detected ellipses data to the selected folder."""
        headers = [
            "major_axis_length",
            "minor_axis_length",
            "equivalent_diameter",
            "area",
            "perimeter",
            "eccentricity",
        ]

        # Save the CSV data
        rows = []
        for image in self.ellipses_properties:
            for circle in image:
                rows.append(
                    [
                        circle["major_axis_length"],
                        circle["minor_axis_length"],
                        circle["equivalent_diameter"],
                        circle["area"],
                        circle["perimeter"],
                        circle["eccentricity"],
                    ]
                )

        # Write the data into a CSV file
        with open(export_path, mode="w", newline="") as data_file:
            writer = csv.writer(data_file)

            # Write the header
            writer.writerow(headers)

            # Write the rows of data
            writer.writerows(rows)

    def save_config_data(self, export_path: str) -> None:
        """Save the configuration data to a txt file."""
        # Store the segmentation data
        headers_seg: list[str] = []
        rows_seg: list[str] = []
        for algorithm_name, params in self.all_methods_n_params.items():
            if algorithm_name == self.algorithm:
                for key, value in params.items():
                    headers_seg.append(key)
                    rows_seg.append(cast(str, value))

        # Store the Filtering Data
        headers_1: list[str] = []
        rows_1: list[str] = []
        headers_2: list[str] = []
        rows_2: list[str] = []
        for key, value in self.param_dict_1.items():  # type: ignore
            headers_1.append(key)
            rows_1.append(cast(str, value))

        if_find_circles: bool = self.param_dict_1.get("find_circles(Y/N)") == "Y"
        if self.param_dict_2.get("find_circles(Y/N)") == "Y":
            if_find_circles = True
            for key, value in self.param_dict_2.items():  # type: ignore
                headers_2.append(key)
                rows_2.append(cast(str, value))

        with open(export_path, mode="w", newline="") as data_file:
            writer = csv.writer(data_file)

            # Write algorithm name and parameters
            writer.writerow(["Segmentation Parameters"])
            writer.writerow(["Algorithm", self.algorithm])
            for i in range(len(headers_seg)):
                writer.writerow([headers_seg[i], rows_seg[i]])
            writer.writerow([])  # Empty row for separation

            # Write the first set of parameters (header, value pairs)
            writer.writerow(["Filtering Parameters"])
            for i in range(len(headers_1)):
                writer.writerow([headers_1[i], rows_1[i]])

            # If find circles is enabled, write the second set of parameters
            if if_find_circles:
                writer.writerow([])  # Empty row for separation
                writer.writerow(["Circle Detection Parameters"])
                for i in range(len(headers_2)):
                    writer.writerow([headers_2[i], rows_2[i]])

            # Add timestamp
            writer.writerow([])
            writer.writerow(["Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            logging.info(f"Configuration data saved to {export_path}")

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class MainHandler:
    """Main controller class that coordinates all handlers and GUI components.

    This class serves as the central coordinator for the application, initializing and
    managing all handler classes, models, and the GUI. It connects signals between
    components and provides methods to handle user interactions across different tabs.

    Attributes:
        toml_file_path (Path): Path to the configuration TOML file.
        toml_handler (TomlFileHandler): Handler for loading and validating TOML configuration.
        input_file_model (InputFilesModel): Model for managing input image files.
        folder_tab_handler (FolderTabHandler): Handler for the folder selection tab.
        calibration_model (CalibrationModel): Model for managing calibration data.
        calibration_tab_handler (CalibrationTabHandler): Handler for the calibration tab.
        image_processing_model (ImageProcessingModel): Model for image processing operations.
        image_processing_tab_handler (ImageProcessingTabHandler): Handler for the image processing tab.
        results_tab_handler (ResultsTabHandler): Handler for the results tab.
        app (QApplication): The main Qt application instance.
        gui (MainWindow): The main GUI window instance.
        export_handler (ExportSettingsHandler): Handler for export settings.
    """

    def __init__(self) -> None:
        """Initialize the MainHandler with all necessary components and handlers.

        Sets up all models, handlers, and GUI components, and establishes connections
        between them to enable proper application functionality.
        """
        # First try to find config.toml relative to the executable when packaged
        import os
        import sys

        self.gui: MainWindow

        # Set up logging to capture terminal output
        self.setup_logging()

        # Get the base directory for the application
        if getattr(sys, "frozen", False):
            # If the application is run as a bundle (PyInstaller)
            base_dir = os.path.dirname(sys.executable)
            # Try to find config.toml in the same directory as the executable
            config_path = os.path.join(base_dir, "config.toml")
            if os.path.exists(config_path):
                self.toml_file_path = Path(config_path)
            else:
                # Fall back to the bundled path
                self.toml_file_path = Path(__file__).parent.parent / "config.toml"
        else:
            # If running in development mode
            self.toml_file_path = Path(__file__).parent.parent / "config.toml"

        logging.info(f"Using TOML file at: {self.toml_file_path}")

        self.initialize_gui()
        self.initialize_handlers()
        self.initialize_handlers_signals()
        self.initialize_new_export_settings()
        self.load_export_settings()

        self.load_gui_for_handlers()
        self.connect_gui_and_handlers()
        self.gui_exiting()

    def initialize_handlers(self) -> None:
        """Initialize all handler classes and models used by the application.

        Creates instances of all necessary handlers and models with appropriate
        configuration parameters from the TOML file.
        """
        logging.info("Initializing Handlers...")
        self.toml_handler = TomlFileHandler(self.toml_file_path)

        self.input_file_model = InputFilesModel()
        self.folder_tab_handler = FolderTabHandler(self.input_file_model, params=self.toml_handler.params)

        self.calibration_model = CalibrationModel()
        self.calibration_tab_handler = CalibrationTabHandler(self.calibration_model, params=self.toml_handler.params)

        self.image_processing_model = ImageProcessingModel(params=self.toml_handler.params)
        self.image_processing_tab_handler = ImageProcessingTabHandler(
            self.image_processing_model, params=self.toml_handler.params
        )

        self.results_tab_handler = ResultsTabHandler(params=self.toml_handler.params)

    def initialize_handlers_signals(self) -> None:
        """Connect signals between handlers to enable communication.

        Establishes signal-slot connections between different handlers to enable
        proper event propagation and response to user actions.
        """
        self.image_processing_tab_handler.batch_processing_done.connect(self.start_generate_histogram)
        self.image_processing_tab_handler.check_for_export_path.connect(self.check_before_batch)
        self.results_tab_handler.check_for_export_path.connect(self.check_before_saving_results)

    def initialize_gui(self) -> None:
        """Initialize the main GUI application and window, and display it.

        Creates the QApplication instance and the main window for the application.
        """
        from bubble_analyser.gui import MainWindow

        logging.basicConfig(level=logging.INFO)
        logging.info("Initializing GUI...")

        self.app = QApplication(sys.argv)
        self.gui = MainWindow()
        self.gui.show()

    def gui_exiting(self) -> None:
        """Handle the exit of the GUI application.

        Ensures that the application exits gracefully when the main window is closed.
        """
        sys.exit(self.app.exec())

    def load_gui_for_handlers(self) -> None:
        """Load GUI references into all handlers.

        Provides each handler with a reference to the main GUI instance to enable
        direct interaction with GUI components. This method must be called after
        the GUI has been initialized and before handlers start interacting with
        GUI components.
        """
        logging.info("Connecting GUI and Handlers...")
        self.folder_tab_handler.load_gui(self.gui)
        self.calibration_tab_handler.load_gui(self.gui)
        self.image_processing_tab_handler.load_gui(self.gui)
        self.results_tab_handler.load_gui(self.gui)

    def connect_gui_and_handlers(self) -> None:
        """Connect GUI components to their respective handlers.

        Sets up signal-slot connections between GUI components and their respective
        handlers to enable proper event handling and interaction.
        """
        # menubar
        self.gui.export_setting_action.triggered.connect(self.menubar_open_export_settings_dialog)
        self.gui.restart_action.triggered.connect(self.menubar_ask_if_restart)

        # folder tab
        self.gui.folder_path_edit.setText(str(self.folder_tab_handler.image_path))
        self.gui.select_folder_button.clicked.connect(self.tab1_select_folder)
        self.gui.confirm_folder_button.clicked.connect(self.tab1_confirm_folder_selection)
        self.gui.image_list.clicked.connect(self.folder_tab_handler.preview_image_folder_tab)

        # calibration tab
        self.gui.pixel_img_name.setText(str(self.calibration_tab_handler.px_img_path))
        self.gui.pixel_img_select_button.clicked.connect(self.tab2_select_ruler_button)
        self.gui.bg_corr_select_button.clicked.connect(self.tab2_select_bg_corr_image)
        self.gui.confirm_px_mm_button.clicked.connect(
            self.tab2_confirm_calibration
        )  # Connect confirm button to the handler

        # image processing tab
        # column 1
        self.gui.prev_button.clicked.connect(lambda: self.tab3_update_sample_image("prev"))
        self.gui.next_button.clicked.connect(lambda: self.tab3_update_sample_image("next"))
        self.gui.preview_processed_images_button.clicked.connect(self.tab3_preview_processed_images)
        # column 2
        self.image_processing_tab_handler.initialize_algorithm_combo()
        self.gui.algorithm_combo.currentTextChanged.connect(
            lambda: self.tab3_handle_algorithm_change(self.gui.algorithm_combo.currentText())
        )
        self.tab3_load_parameter_table_1(self.gui.algorithm_combo.currentText())
        self.gui.preview_button1.clicked.connect(self.tab3_confirm_parameter_before_filtering)
        # column 3
        self.tab3_initialize_parameter_table_2()
        self.gui.fc_checkbox.stateChanged.connect(self.tab3_handle_find_circles)
        self.gui.manual_adjustment_button.clicked.connect(self.tab3_ellipse_manual_adjustment)
        self.gui.preview_button2.clicked.connect(self.tab3_confirm_parameter_for_filtering)
        self.gui.batch_process_button.clicked.connect(self.tab3_ask_if_batch)

    def disconnect_gui_and_handlers(self) -> None:
        """Disconnect GUI components from their respective handlers.

        Removes signal-slot connections between GUI components and their respective
        handlers to prevent further event handling and interaction.
        """
        # menubar
        self.gui.export_setting_action.triggered.disconnect(self.menubar_open_export_settings_dialog)
        self.gui.restart_action.triggered.disconnect(self.menubar_ask_if_restart)

        # folder tab
        self.gui.folder_path_edit.setText(str(self.folder_tab_handler.image_path))
        self.gui.select_folder_button.clicked.disconnect(self.tab1_select_folder)
        self.gui.confirm_folder_button.clicked.disconnect(self.tab1_confirm_folder_selection)
        self.gui.image_list.clicked.disconnect(self.folder_tab_handler.preview_image_folder_tab)

        # calibration tab
        self.gui.pixel_img_name.setText(str(self.calibration_tab_handler.px_img_path))
        self.gui.pixel_img_select_button.clicked.disconnect(self.tab2_select_ruler_button)
        self.gui.bg_corr_select_button.clicked.disconnect(self.tab2_select_bg_corr_image)
        self.gui.confirm_px_mm_button.clicked.disconnect(
            self.tab2_confirm_calibration
        )  # Connect confirm button to the handler

        # image processing tab
        # column 1
        self.gui.prev_button.clicked.disconnect()
        self.gui.next_button.clicked.disconnect()
        self.gui.preview_processed_images_button.clicked.disconnect(self.tab3_preview_processed_images)
        # column 2
        self.image_processing_tab_handler.initialize_algorithm_combo()
        self.gui.algorithm_combo.currentTextChanged.disconnect()
        self.tab3_load_parameter_table_1(self.gui.algorithm_combo.currentText())
        self.gui.preview_button1.clicked.disconnect(self.tab3_confirm_parameter_before_filtering)
        # column 3
        self.tab3_initialize_parameter_table_2()
        self.gui.fc_checkbox.stateChanged.disconnect(self.tab3_handle_find_circles)
        self.gui.manual_adjustment_button.clicked.disconnect(self.tab3_ellipse_manual_adjustment)
        self.gui.preview_button2.clicked.disconnect(self.tab3_confirm_parameter_for_filtering)
        self.gui.batch_process_button.clicked.disconnect(self.tab3_ask_if_batch)

        # results tab
        self.gui.pdf_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.cdf_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.bins_spinbox.valueChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.min_x_axis_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.max_x_axis_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.legend_position_combobox.currentIndexChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.d32_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dmean_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dxy_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dxy_x_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dxy_y_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.save_button.clicked.disconnect()

        self.image_processing_tab_handler.batch_processing_done.disconnect(self.start_generate_histogram)
        self.image_processing_tab_handler.check_for_export_path.disconnect(self.check_before_batch)
        self.results_tab_handler.check_for_export_path.disconnect(self.check_before_saving_results)

    def disconnect_handlers_signals(self) -> None:
        """Disconnect signals between handlers to prevent further event handling.

        Removes signal-slot connections between handlers to prevent further event
        handling and interaction.
        """
        self.image_processing_tab_handler.batch_processing_done.disconnect()

    def clear_all_gui_contents(self) -> None:
        """Clear all contents from the GUI components.

        Clears all contents from all GUI components to reset the application state.
        """
        self.gui.image_list.clear()

        self.gui.pixel_img_preview.clear()
        self.gui.bg_corr_image_preview.clear()
        self.gui.manual_px_mm_input.clear()

        self.gui.sample_image_preview.clear()
        self.gui.label_before_filtering.axes.clear()
        self.gui.label_before_filtering.draw()
        self.gui.processed_image_preview.axes.clear()
        self.gui.processed_image_preview.draw()

        self.gui.histogram_canvas.axes.clear()
        self.gui.histogram_canvas.draw()

    def restart(self) -> None:
        """Restart the application.

        Resets the application state and restarts the GUI.
        """
        logging.info("##############################Restarting Mission...##############################")
        self.disconnect_gui_and_handlers()
        self.disconnect_handlers_signals()
        self.clear_all_gui_contents()

        self.initialize_handlers()
        self.initialize_handlers_signals()
        self.load_export_settings()
        self.load_gui_for_handlers()
        self.connect_gui_and_handlers()

        self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.folder_tab))

        logging.info("Application restarted, a new mission initialzed.")
        logging.info("##############################New mission started##############################")

    def load_export_settings(self) -> None:
        """Initialize and configure the export settings handler.

        Creates the export settings handler and provides it to relevant tab handlers
        that need access to export functionality.
        """
        logging.info("Connecting Export Settings with handlers...")
        self.image_processing_tab_handler.export_handler = self.export_handler
        self.results_tab_handler.export_handler = self.export_handler

    def initialize_new_export_settings(self) -> None:
        """Initialize and configure the export settings handler."""
        logging.info("Initializing Export Settings...")
        self.export_handler = ExportSettingsHandler(parent=self.gui)

    def check_before_batch(self) -> None:
        """Check if batch processing can proceed."""
        if self.image_processing_tab_handler.if_save_processed_images:
            if self.check_if_export_settings_loaded():
                self.image_processing_tab_handler.batch_process_images()
            else:
                return
        else:
            self.image_processing_tab_handler.batch_process_images()

    def check_before_saving_results(self) -> None:
        """Check if saving results can proceed."""
        if self.check_if_export_settings_loaded():
            self.results_tab_handler.save_results()
        else:
            return

    def check_if_export_settings_loaded(self) -> bool:
        """Check if export settings have been loaded.

        Returns:
            bool: True if export settings have been loaded, False otherwise.
        """
        if not self.export_handler.if_save_path:
            self._show_warning(
                "Export Path Not Configured",
                "Please finish export settings first from menu bar (settings -> export setting).",
            )
            logging.info("Export Settings not loaded.")
            return False
        else:
            logging.info("Export Settings loaded.")
            return True

    def menubar_open_export_settings_dialog(self) -> None:
        """Open the export settings dialog from the menu bar.

        Displays a dialog allowing the user to configure export settings for
        processed images and results.
        """
        self.export_handler.exec()

    def menubar_ask_if_restart(self) -> None:
        """Show a confirmation dialog for restarting the application.

        Displays a message box asking the user if they want to restart the application,
        warning that unsaved progress will be lost.
        """
        restart = QMessageBox.question(
            self.gui,
            "Restart",
            "Do you want to start a new mission? \n Current unsaved progress will be lost",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if restart == QMessageBox.StandardButton.Yes:
            self.restart()

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)

    def tab1_select_folder(self) -> None:
        """Handle folder selection in the first tab.

        Delegates the folder selection process to the folder tab handler.
        """
        self.folder_tab_handler.select_folder()

    def tab1_confirm_folder_selection(self) -> None:
        """Confirm the folder selection in the first tab.

        Confirms the selected folder and loads the image path list into the
        image processing model.
        """
        self.folder_tab_handler.confirm_folder_selection()
        # load image path list to Processing Hanlder
        self.image_processing_model.confirm_folder_selection(self.input_file_model.image_list_full_path_in_path)

    def tab2_get_px2mm_ratio(self) -> None:
        """Calculate the pixel-to-millimeter ratio in the calibration tab.

        Delegates the pixel-to-millimeter calculation to the calibration tab handler.
        """
        self.calibration_tab_handler.get_px2mm_ratio()

    def tab2_select_ruler_button(self) -> None:
        """Select a ruler image for pixel-to-millimeter calibration.

        Delegates the ruler image selection to the calibration tab handler.
        """
        self.calibration_tab_handler.select_ruler_button()

    def tab2_select_bg_corr_image(self) -> None:
        """Select a background correction image in the calibration tab.

        Delegates the background image selection to the calibration tab handler.
        """
        self.calibration_tab_handler.select_bg_corr_image()

    def tab2_confirm_calibration(self) -> None:
        """Confirm the calibration settings and update models.

        Confirms the calibration settings and updates the image processing model
        with the calibration data, including pixel-to-millimeter ratio and
        background image if selected.
        """
        self.calibration_tab_handler.confirm_calibration()
        self.image_processing_model.update_px2mm(self.calibration_model.px2mm)

        if self.calibration_model.if_bknd:
            self.image_processing_model.if_bknd = self.calibration_model.if_bknd
            self.image_processing_model.get_bknd_img_path(self.calibration_model.bknd_img_path)

    def tab3_load_parameter_table_1(self, algorithm: str) -> None:
        """Load parameters for the selected algorithm in the first parameter table.

        Args:
            algorithm (str): The name of the selected algorithm.
        """
        self.image_processing_tab_handler.load_parameter_table_1(algorithm)

    def tab3_initialize_parameter_table_2(self) -> None:
        """Initialize the second parameter table in the image processing tab.

        Prepares the table for displaying filtering parameters.
        """
        self.image_processing_tab_handler.initialize_parameter_table_2()

    def tab3_update_sample_image(self, status: str) -> None:
        """Update the sample image display based on processing status.

        Args:
            status (str): The current processing status to determine which image to display.
        """
        self.image_processing_tab_handler.update_sample_image(status)

    def tab3_preview_processed_images(self) -> None:
        """Preview the processed images in the image processing tab.

        Delegates the image preview functionality to the image processing tab handler.
        """
        self.image_processing_tab_handler.preview_processed_images()

    def tab3_handle_algorithm_change(self, algorithm: str) -> None:
        """Handle a change in the selected algorithm in the image processing tab.

        Delegates the algorithm change handling to the image processing tab handler,
        which updates parameters and UI elements accordingly.

        Args:
            algorithm (str): The name of the newly selected algorithm.
        """
        self.image_processing_tab_handler.handle_algorithm_change(algorithm)

    def tab3_confirm_parameter_before_filtering(self) -> None:
        """Confirm the parameters for the first step of image processing.

        Delegates the parameter confirmation to the image processing tab handler,
        which validates all parameters against the configuration schema before
        proceeding with the first processing step.
        """
        self.image_processing_tab_handler.confirm_parameter_before_filtering()

    def tab3_handle_find_circles(self) -> None:
        """Handle the state change of the "Find Circles" checkbox in the image processing tab.

        Delegates the checkbox state change handling to the image processing tab handler,
        which updates the UI and processing state accordingly.

        Args:
            state: The new state of the checkbox.
        """
        self.image_processing_tab_handler.handle_find_circles()

    def tab3_confirm_parameter_for_filtering(self) -> None:
        """Confirm the filtering parameters and apply them to the current image.

        Delegates the filtering parameter confirmation to the image processing tab handler,
        which validates all filtering parameters against the configuration schema before
        proceeding with the second processing step.
        """
        self.image_processing_tab_handler.confirm_parameter_for_filtering()

    def tab3_ellipse_manual_adjustment(self) -> None:
        """Enable manual adjustment of detected ellipses in the image processing tab.

        Delegates the ellipse manual adjustment functionality to the image processing
        tab handler, allowing users to modify automatically detected ellipses.
        """
        self.image_processing_tab_handler.ellipse_manual_adjustment()

    def tab3_ask_if_batch(self) -> None:
        """Ask the user if they want to perform batch processing on all images.

        Delegates the batch processing confirmation dialog to the image processing
        tab handler, which prompts the user and initiates batch processing if confirmed.
        """
        self.image_processing_tab_handler.ask_if_batch()

    def start_generate_histogram(self) -> None:
        """Generate histograms based on the processed ellipse properties.

        Loads the ellipse properties from the image processing model into the
        results tab handler and triggers histogram generation to visualize
        the distribution of bubble sizes and other properties.
        """
        self.results_tab_handler.load_ellipse_properties(
            self.image_processing_model.ellipses_properties,
            self.image_processing_model.algorithm,
            self.image_processing_model.all_methods_n_params,
            self.image_processing_model.filter_param_dict_1,
            self.image_processing_model.filter_param_dict_2,
        )
        self.results_tab_handler.generate_histogram()

    def setup_logging(self) -> None:
        """Set up logging to capture terminal output to a file.

        This method configures a logging system that captures all print statements
        and other terminal outputs to a timestamped log file in the 'logs' directory.
        """
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create a timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"bubble_analyser_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

        logging.info(f"Starting Bubble Analyser application. Log file: {log_file}")


if __name__ == "__main__":
    main_handler = MainHandler()
