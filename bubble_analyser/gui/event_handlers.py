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

import csv
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import toml as tomllib  # type: ignore
from numpy import typing as npt
from pydantic import ValidationError
from PySide6.QtCore import QProcess, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
)

from bubble_analyser.gui import (
    CalibrationModel,
    ImageProcessingModel,
    InputFilesModel,
    WorkerThread,
)
from bubble_analyser.processing import Config


class ExportSettingsHandler(QDialog):
    """A dialog for configuring export settings for processed images.

    This class provides a user interface for selecting and confirming the directory
    where processed images will be saved.

    Attributes:
        save_path (Path): The directory path where processed images will be saved.
        default_path_edit (QLineEdit): Text field displaying the current save path.
        confirm_button (QPushButton): Button to confirm the selected path.
    """

    def __init__(self, parent=None, params: Config = None) -> None:  # type: ignore
        """Initialize the export settings dialog.

        Args:
            parent: The parent widget. Defaults to None.
            params (Config, optional): Configuration parameters containing the default save path. Defaults to None.
        """
        super().__init__(parent)

        self.setWindowTitle("Export Settings")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout(self)

        self.save_path: Path = params.save_path

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

        layout.addLayout(path_layout)

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

    def select_folder(self) -> None:
        """Open a file dialog to select a folder for saving processed images.

        Updates the text field with the selected folder path.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
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

    def load_gui(self, gui) -> None:  # type: ignore
        """Load a reference to the GUI instance for displaying warnings.

        This method should be called after the GUI has been initialized.
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

    # def check_params(self, dict_params: dict) -> bool:
    #     try:
    #         Config(**dict_params)
    #     except ValidationError as e:
    #         error_str = str(e)
    #         self._show_warning("Error in Config File Setting", error_str)
    #     return True

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

    def load_gui(self, gui) -> None:  # type: ignore
        """Load a reference to the GUI instance.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def select_folder(self) -> None:
        """Handle the folder selection process.

        Opens a file dialog for the user to select a folder containing images.
        If a folder has already been confirmed, displays a warning instead.
        """
        if self.model.sample_images_confirmed:
            self._show_warning("Selection Locked", "You have already confirmed the folder selection.")
            return

        folder_path = QFileDialog.getExistingDirectory(self.gui, "Select Folder")
        if folder_path:
            self._update_folder_path(folder_path)
            self._populate_image_list(folder_path)

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

        for file_name in images:
            self.gui.image_list.addItem(file_name)

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

    def load_gui(self, gui) -> None:  # type: ignore
        """Load a reference to the GUI instance.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def select_px_mm_image(self) -> None:
        """Handle the selection of a ruler image for pixel-to-millimeter calibration.

        Opens a file dialog for selecting a ruler image and updates the preview in the GUI.
        If calibration has already been confirmed, displays a warning instead.
        """
        if self.calibration_model.calibration_confirmed:
            self._show_warning(
                "Selection Locked",
                "You have already confirmed the ruler image selection.",
            )
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self.gui, "Select Ruler Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
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
            px2mm = self.calibration_model.get_px2mm_ratio(
                pixel_img_path=img_path, img_resample=self.img_resample, gui=self.gui
            )
            self.gui.manual_px_mm_input.setText(f"{px2mm:.3f}")
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

        image_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "Select Background Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
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

    def confirm_calibration(self) -> None:
        """Confirm the calibration settings and proceed to the next tab.

        Updates the calibration model with the final values and switches to the
        image processing tab. If calibration has already been confirmed, displays
        a warning instead.
        """
        if self.calibration_model.calibration_confirmed:
            self.gui.manual_px_mm_input.setText(f"{self.calibration_model.px2mm:.3f}")
            self._show_warning("Selection Locked", "You have already confirmed the calibration.")
            return

        self.calibration_model.bknd_img_path = Path(self.gui.bg_corr_image_name.text())
        self.calibration_model.px2mm = float(self.gui.manual_px_mm_input.text())
        self.calibration_model.confirm_calibration()

        self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.image_processing_tab))

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

    def __init__(self, image_processing_model: ImageProcessingModel, params: Config) -> None:
        """Initialize the image processing tab handler.

        Args:
            image_processing_model (ImageProcessingModel): Model for managing image processing.
            params (Config): Configuration parameters for processing.
        """
        super().__init__()
        self.model: ImageProcessingModel = image_processing_model
        self.params: Config = params
        self.current_index: int = 0
        self.algorithm_list: list[str] = []
        self.export_handler: ExportSettingsHandler
        self.if_save_processed_images = False

        self.save_path: Path = cast(Path, None)

    def load_gui(self, gui) -> None:  # type: ignore
        """Load a reference to the GUI instance.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def pass_filter_params(self, filter_param_dict: dict[str, str | float]) -> None:
        """Pass filter parameters to the processing model.

        Args:
            filter_param_dict (dict[str, int | float]): Dictionary of filter parameters.
        """
        self.model.load_filter_params(filter_param_dict)

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

        Args:
            name (str): The name of the parameter to validate.
            value (int | float): The value to validate.

        Returns:
            bool: True if the parameter is valid, False otherwise.
        """
        if name == "element_size":
            try:
                self.params.element_size = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid Element Size", str(e))
                return False

        if name == "connectivity":
            try:
                self.params.connectivity = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid Connectivity", str(e))
                return False

        if name == "resample":
            try:
                self.params.resample = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Resample Factor", str(e))
                return False

        if name == "max_thresh":
            try:
                self.params.max_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Max Threshold", str(e))
                return False

        if name == "min_thresh":
            try:
                self.params.min_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Threshold", str(e))
                return False

        if name == "step_size":
            try:
                self.params.step_size = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Step Size", str(e))
                return False

        if name == "max_eccentricity":
            try:
                self.params.max_eccentricity = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Max Eccentricity", str(e))
                return False

        if name == "min_solidity":
            try:
                self.params.min_solidity = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Solidity", str(e))
                return False

        if name == "min_size":
            try:
                self.params.min_size = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Size", str(e))
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
        print("(event_handlers/update_sample_img)current index: ", self.current_index)
        self.preview_image()

    # -------Second Column Functions-------------------------
    def initialize_algorithm_combo(self) -> None:
        """Initialize the algorithm combo box with available processing methods.

        Populates the combo box with the names of all available processing algorithms
        and sets the default algorithm.
        """
        # Initialize the algorithm combo box
        # And achieve all the available methods' names

        for algorithm, params in self.model.all_methods_n_params.items():
            print("initialize algorithm:", algorithm)
            self.algorithm_list.append(algorithm)

        self.gui.algorithm_combo.addItems(self.algorithm_list)
        self.update_model_algorithm(self.algorithm_list[0])

    def load_parameter_table_1(self, algorithm: str) -> None:
        """Load the parameter table with values for the selected algorithm.

        Args:
            algorithm (str): The name of the algorithm whose parameters should be loaded.
        """
        # pass the current algorithm text in the gui to the model
        # This function only triggered by first initialization and algorithm change

        self.current_algorithm = algorithm
        print("current algorithm:", self.current_algorithm)

        for algorithm_name, params in self.model.all_methods_n_params.items():
            print("algorithm name:", algorithm_name)
            if algorithm_name == self.current_algorithm:
                self.gui.param_sandbox1.setRowCount(len(params))

                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    self.gui.param_sandbox1.setItem(row, 0, QTableWidgetItem(name))
                    self.gui.param_sandbox1.setItem(row, 1, QTableWidgetItem(str(value)))
                    # self.temp_param_dict[name] = value

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

    def confirm_parameter_before_filtering(self) -> None:
        """Confirm the parameters for the first step of image processing.

        Validates all parameters against the configuration schema before
        proceeding with the first processing step.
        """
        self.update_segment_parameters()
        print("------------------------------Validating Segment Parameters------------------------------")
        # Update the model parameters
        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    print(
                        "Checking steps in confirm_parameter_before_filtering: ",
                        name,
                        value,
                    )
                    if_valid = self.check_params(name, value)
                    if not if_valid:
                        return

        # self.pass_segment_params(self.model.segment_param_dict)
        self._process_step_1()

    def looks_like_float(self, s: str) -> bool:
        """Check if a string represents a floating-point number.

        Args:
            s (str): The string to check.

        Returns:
            bool: True if the string represents a floating-point number, False otherwise.
        """
        try:
            f = float(s)
            # Check if it has a fractional part
            print(s, "is float?")
            return not f.is_integer()
        except ValueError:
            print(s, "is not float")
            return False

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
            return int(value) if value.is_integer() else value
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
        params = self.extract_parameters_from_table(self.gui.param_sandbox1)

        print("------------------------------Updating Parameters------------------------------")

        # Update the model's dictionary for the selected algorithm
        for algorithm_name, params_in_dict in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for name, value in params.items():
                    print("Updating", name, "to", value)
                    params_in_dict[name] = value

        return True

    def _process_step_1(self) -> None:
        """Execute the first step of image processing.

        Calls the model's step_1_main method to process the current image
        and updates the preview with the results.
        """
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
        param_dict = self.model.filter_param_dict

        self.gui.param_sandbox2.setRowCount(len(param_dict))
        row = 0
        for property, value in param_dict.items():
            print("Filter param name:", property, ", value:", value)
            self.gui.param_sandbox2.setItem(row, 0, QTableWidgetItem(property))
            self.gui.param_sandbox2.setItem(row, 1, QTableWidgetItem(str(value)))
            row += 1

        self.filter_param_dict: dict[str, float | str] = param_dict

    def confirm_parameter_for_filtering(self) -> None:
        """Confirm the filtering parameters and apply them to the current image.

        Validates all filtering parameters against the configuration schema before
        proceeding with the second processing step.
        """
        print("------------------------------Updating Filtering Parameters------------------------------")
        self.store_filter_params()

        print("------------------------------Validating Filter Parameters------------------------------")
        for name, value in self.filter_param_dict.items():
            print(
                "Checking steps in confirm_parameter_before_filtering: ",
                name,
                value,
            )
            if_valid = self.check_params(name, value)
            if not if_valid:
                return

        self.pass_filter_params(self.filter_param_dict)
        self._process_step_2()

    def store_filter_params(self) -> None:
        """Store the filtering parameters from the GUI table into the temporary dictionary.

        Extracts the filtering parameters from the table widget and updates the
        temporary filter parameter dictionary with the new values.
        """
        params: dict[str, float | str] = {}

        for row in range(self.gui.param_sandbox2.rowCount()):
            name_item = self.gui.param_sandbox2.item(row, 0)
            value_item = self.gui.param_sandbox2.item(row, 1)
            if name_item and value_item:
                param_name = name_item.text()
                param_value = value_item.text()

                if param_name == "find_circles(Y/N)":
                    # Handle string parameter
                    params[param_name] = str(param_value)
                    self.filter_param_dict[param_name] = str(param_value)
                    print(param_name, param_value)
                else:
                    # Handle numeric parameter
                    float_value = cast(float, param_value)
                    params[param_name] = float_value
                    self.filter_param_dict[param_name] = float_value

                print("Updating", param_name, "to", param_value)

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
        confirm_dialog = self.create_confirm_dialog()
        self.create_save_images_checkbox(confirm_dialog)

        response = confirm_dialog.exec()

        if response == QMessageBox.StandardButton.Ok:
            self.batch_process_images()
        else:
            print("Batch processing canceled.")

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
        print("------------------------------Validating Segment Parameters------------------------------")
        # Update the model parameters
        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    print(
                        "Checking steps in confirm_parameter_before_filtering: ",
                        name,
                        value,
                    )
                    if_valid = self.check_params(name, value)
                    if not if_valid:
                        return

        print("------------------------------Validating Filter Parameters------------------------------")
        self.store_filter_params()
        self.update_segment_parameters()
        self.pass_filter_params(self.filter_param_dict)
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
        print("update progress bar:", value)

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


class ResultsTabHandler:
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

    def __init__(self, params: Config) -> None:
        """Initialize the results tab handler.

        Args:
            params (Config): Configuration parameters containing default values.
        """
        # self.gui = gui
        self.params = params
        self.save_path = params.save_path

        self.ellipses_properties: list[list[dict[str, float]]]
        self.export_handler: ExportSettingsHandler

    def load_gui(self, gui) -> None:  # type: ignore
        """Load a reference to the GUI instance.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def load_ellipse_properties(self, properties: list[list[dict[str, float]]]) -> None:
        """Load the properties of detected ellipses for display and analysis.

        Args:
            properties (list[list[dict[str, float]]]): Properties of detected ellipses for all images.
        """
        self.ellipses_properties = properties
        pass

    def generate_histogram(self) -> None:
        """Generate and display a histogram of bubble sizes.

        Creates a histogram showing the distribution of equivalent diameters of detected bubbles.
        Optionally displays PDF, CDF, and characteristic diameters (d32, dmean, dxy) based on
        user selections. Updates the plot with appropriate labels and legend.
        """
        num_bins = self.gui.bins_spinbox.value()
        show_pdf = self.gui.pdf_checkbox.isChecked()
        show_cdf = self.gui.cdf_checkbox.isChecked()
        show_d32 = self.gui.d32_checkbox.isChecked()
        show_dmean = self.gui.dmean_checkbox.isChecked()
        show_dxy = self.gui.dxy_checkbox.isChecked()

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

        # Plot histogram
        counts, bins, patches = self.gui.histogram_canvas.axes.hist(
            equivalent_diameters_array, bins=num_bins, range=(x_min, x_max)
        )
        # Set graph labels
        self.gui.histogram_canvas.axes.set_xlabel("Equivalent diameter [mm]")
        self.gui.histogram_canvas.axes.set_ylabel("Count [#]")

        # Calculate descriptive sizes
        d32, d_mean, dxy = self.calculate_descriptive_sizes(equivalent_diameters_array)

        # Update descriptive size label
        desc_text = f"Results:\nd32 = {d32:.2f} mm\ndmean = {d_mean:.2f} mm\ndxy = {dxy:.2f} mm"
        self.gui.descriptive_size_label.setText(desc_text)

        # Optionally add CDF
        if show_pdf or show_cdf:
            self.gui.histogram_canvas.axes2 = self.gui.histogram_canvas.axes.twinx()
            self.gui.histogram_canvas.axes2.set_ylabel("Probability [%]")

            if show_cdf:
                cdf = np.cumsum(counts) / np.sum(counts) * 100
                self.gui.histogram_canvas.axes2.plot(bins[:-1], cdf, "r-", marker="o", label="CDF")

            if show_pdf:
                pdf = counts / np.sum(counts) * 100
                self.gui.histogram_canvas.axes2.plot(bins[:-1], pdf, "b-", marker="o", label="PDF")

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

        print("legend_position:", legend_position)
        print(legend_location_map.get(legend_position, "upper right"))

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
        # folder_path = self.gui.save_folder_edit.text()
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

        # Assuming `self.histogram_canvas` is a matplotlib canvas
        self.gui.histogram_canvas.fig.savefig(graph_path)

        headers = [
            "major_axis_lengthminor_axis_lengthequivalent_diameter",
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
        with open(csv_path, mode="w", newline="") as data_file:
            writer = csv.writer(data_file)

            # Write the header
            writer.writerow(headers)

            # Write the rows of data
            writer.writerows(rows)

        self._show_warning("Results Saved", f"Results have been saved successfully to {folder_path}.")
        return

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
                base_dir = os.path.dirname(sys.executable)
                self.toml_file_path = Path(os.path.join(base_dir, "bubble_analyser", "gui", "config.toml"))
        else:
            # If running in development mode
            self.toml_file_path = Path("bubble_analyser/gui/config.toml")

        self.initialize_handlers()
        self.initialize_handlers_signals()

        self.initialize_gui()
        self.load_gui_for_handlers()

        self.load_export_settings()
        self.load_full_gui()

    def initialize_handlers(self) -> None:
        """Initialize all handler classes and models used by the application.

        Creates instances of all necessary handlers and models with appropriate
        configuration parameters from the TOML file.
        """
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

    def initialize_gui(self) -> None:
        """Initialize the main GUI application and window.

        Creates the QApplication instance and the main window for the application.
        """
        from bubble_analyser.gui import MainWindow

        self.app = QApplication(sys.argv)
        self.gui: MainWindow = MainWindow(self)

    def load_full_gui(self) -> None:
        """Load and display the complete GUI.

        Finalizes GUI initialization, displays the main window, and starts the
        application event loop.
        """
        self.gui.load_full_gui()
        self.gui.show()
        sys.exit(self.app.exec())

    def load_gui_for_handlers(self) -> None:
        """Load GUI references into all handlers.

        Provides each handler with a reference to the main GUI instance to enable
        direct interaction with GUI components.
        """
        self.folder_tab_handler.load_gui(self.gui)
        self.calibration_tab_handler.load_gui(self.gui)
        self.image_processing_tab_handler.load_gui(self.gui)
        self.results_tab_handler.load_gui(self.gui)

    def load_export_settings(self) -> None:
        """Initialize and configure the export settings handler.

        Creates the export settings handler and provides it to relevant tab handlers
        that need access to export functionality.
        """
        self.export_handler = ExportSettingsHandler(parent=self.gui, params=self.toml_handler.params)
        self.image_processing_tab_handler.export_handler = self.export_handler
        self.results_tab_handler.export_handler = self.export_handler

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

    def tab2_select_px_mm_image(self) -> None:
        """Select a ruler image for pixel-to-millimeter calibration.

        Delegates the ruler image selection to the calibration tab handler.
        """
        self.calibration_tab_handler.select_px_mm_image()

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
        self.results_tab_handler.load_ellipse_properties(self.image_processing_model.ellipses_properties)
        self.results_tab_handler.generate_histogram()

    def save_results(self) -> None:
        """Save the analysis results to disk.

        Delegates the result saving functionality to the results tab handler,
        which exports the processed data according to the configured export settings.
        """
        self.results_tab_handler.save_results()

    def restart(self) -> None:
        """Restart the application by launching a new instance and closing the current one.

        Creates a new detached process running the same Python executable with the same
        arguments, then closes the current application instance.
        """
        QProcess.startDetached(sys.executable, sys.argv)  # type: ignore
        QApplication.quit()

    # def reinitialize_main_handler(self) -> None:
    #     self.__init__()


if __name__ == "__main__":
    main_handler = MainHandler()
