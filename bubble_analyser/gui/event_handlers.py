import csv
import os
import sys
from pathlib import Path

import numpy as np
import toml as tomllib # type: ignore
from numpy import typing as npt
from typing import cast
from pydantic import ValidationError
from PySide6.QtCore import QProcess, Qt, QThread, QTimer, Signal
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
    WorkerThread
)

# from . import component_handlers as ch
from bubble_analyser.gui.component_handlers import * 
from bubble_analyser.processing import Config


class ExportSettingsHandler(QDialog):
    def __init__(self, parent = None, params: Config = None) -> None: # type: ignore
        super().__init__(parent)

        self.setWindowTitle("Export Settings")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout(self)

        self.save_path: Path = params.save_path

        # Default path for results saving
        self.default_path_edit = QLineEdit()
        self.default_path_edit.setPlaceholderText(str(self.save_path)) # type: ignore
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
        if self.check_if_path_valid() is False: 
            return None
        super().accept()
        self.save_path = self.default_path_edit.text()

    def select_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.default_path_edit.setText(folder_path)

    def check_if_path_valid(self) -> bool:
        try:
            Path(self.save_path).resolve()
            return True
        except FileNotFoundError as e:
            error_str = str(e)
            QMessageBox.warning(self, "Error", error_str)
            return False


class TomlFileHandler:
    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self.params: Config
        self.load_toml()

    def load_gui(self) -> None:
        self.gui = gui # type: ignore

    def load_toml(self) -> None:
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
        QMessageBox.warning(self.gui, title, message) 


class FolderTabHandler:
    def __init__(self, model: InputFilesModel, params: Config) -> None:
        self.model: InputFilesModel = model
        self.image_path: Path = params.raw_img_path

    def load_gui(self, gui) -> None: # type: ignore
        self.gui = gui

    def select_folder(self) -> None:
        if self.model.sample_images_confirmed:
            self._show_warning(
                "Selection Locked", "You have already confirmed the folder selection."
            )
            return

        folder_path = QFileDialog.getExistingDirectory(self.gui, "Select Folder")
        if folder_path:
            self._update_folder_path(folder_path)
            self._populate_image_list(folder_path)

    def _update_folder_path(self, folder_path: str) -> None:
        """Update the model and GUI with the selected folder path."""
        self.model.folder_path = cast(Path, folder_path)
        self.gui.folder_path_edit.setText(folder_path)

    def _populate_image_list(self, folder_path: str) -> None:
        images, _ = self.model.get_image_list(folder_path)
        self.gui.image_list.clear()

        for file_name in images:
            self.gui.image_list.addItem(file_name)

        self.gui.image_list.addItems(images)

    def confirm_folder_selection(self) -> None:
        if self.model.sample_images_confirmed:
            self._show_warning(
                "Selection Locked", "You have already confirmed the folder selection."
            )
            return

        folder_path = self.gui.folder_path_edit.text()
        if folder_path:
            self._update_folder_path(folder_path)
            self._populate_image_list(folder_path)
            self.model.confirm_folder_selection(folder_path)
            self.gui.tabs.setCurrentIndex(
                self.gui.tabs.indexOf(self.gui.calibration_tab)
            )

    def preview_image_folder_tab(self) -> None:
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
        QMessageBox.warning(self.gui, title, message)


class CalibrationTabHandler:
    def __init__(self, calibration_model: CalibrationModel, params: Config) -> None:
        self.calibration_model: CalibrationModel = calibration_model
        self.img_resample: float = params.resample
        self.px_img_path: Path = params.ruler_img_path

    def load_gui(self, gui) -> None: # type: ignore
        self.gui = gui

    def select_px_mm_image(self) -> None:
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
            # self.calibration_model.pixel_img_path = Path(image_path)
            self.gui.pixel_img_name.setText(image_path)
            pixmap = QPixmap(image_path)
            self.gui.pixel_img_preview.setPixmap(
                pixmap.scaled(
                    self.gui.pixel_img_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )

    def get_px2mm_ratio(self) -> None:
        if self.calibration_model.calibration_confirmed:
            self._show_warning(
                "Selection Locked", "You have already confirmed the pixel-to-mm ratio."
            )
            return

        img_path: Path = cast(Path, self.gui.pixel_img_name.text())
        if os.path.exists(img_path):
            # self.calibration_model.pixel_img_path = img_path
            px2mm = self.calibration_model.get_px2mm_ratio(
                pixel_img_path=img_path, img_resample=self.img_resample, gui=self.gui
            )
            self.gui.manual_px_mm_input.setText(f"{px2mm:.3f}")
        else:
            self.gui.statusBar().showMessage(
                "Image file does not exist or not selected.", 5000
            )

    def select_bg_corr_image(self) -> None:
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
        if self.calibration_model.calibration_confirmed:
            self.gui.manual_px_mm_input.setText(f"{self.calibration_model.px2mm:.3f}")
            self._show_warning(
                "Selection Locked", "You have already confirmed the calibration."
            )
            return

        self.calibration_model.bknd_img_path = Path(self.gui.bg_corr_image_name.text())
        self.calibration_model.px2mm = float(self.gui.manual_px_mm_input.text())
        self.calibration_model.confirm_calibration()

        self.gui.tabs.setCurrentIndex(
            self.gui.tabs.indexOf(self.gui.image_processing_tab)
        )

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self.gui, title, message)


class ImageProcessingTabHandler(QThread):
    batch_processing_done = Signal()

    def __init__(
        self, image_processing_model: ImageProcessingModel, params: Config
    ) -> None:
        super().__init__()
        self.model: ImageProcessingModel = image_processing_model
        self.params: Config = params
        self.current_index: int = 0
        self.algorithm_list: list[str] = []
        self.export_handler: ExportSettingsHandler
        self.if_save_processed_images = False

        self.temp_param_dict: dict[str, int | float] = {}
        self.temp_filter_param_dict = {
            "max_eccentricity": params.max_eccentricity,
            "min_solidity": params.min_solidity,
            "min_size": params.min_size,
        }

        self.save_path: Path = cast(Path, None)

    def load_gui(self, gui) -> None: # type: ignore
        self.gui = gui

    def pass_filter_params(self, filter_param_dict: dict[str, int | float]) -> None:
        self.model.load_filter_params(filter_param_dict)

    def preview_image(self) -> None:
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

    def check_params(self, name: str, value: int | float) -> bool:
        if name == "element_size":
            try:
                self.params.element_size = value
            except ValidationError as e:
                self._show_warning("Invalid Element Size", str(e))
                return False

        if name == "connectivity":
            try:
                self.params.connectivity = value
            except ValidationError as e:
                self._show_warning("Invalid Connectivity", str(e))
                return False

        if name == "threshold_value":
            try:
                self.params.threshold_value = value
            except ValidationError as e:
                self._show_warning("Invalid Threshold Value", str(e))
                return False

        if name == "resample":
            try:
                self.params.resample = value
            except ValidationError as e:
                self._show_warning("Invalid Resample Factor", str(e))
                return False

        if name == "max_thresh":
            try:
                self.params.max_thresh = value
            except ValidationError as e:
                self._show_warning("Invalid Max Threshold", str(e))
                return False

        if name == "min_thresh":
            try:
                self.params.min_thresh = value
            except ValidationError as e:
                self._show_warning("Invalid Min Threshold", str(e))
                return False

        if name == "step_size":
            try:
                self.params.step_size = value
            except ValidationError as e:
                self._show_warning("Invalid Step Size", str(e))
                return False

        if name == "max_eccentricity":
            try:
                self.params.max_eccentricity = value
            except ValidationError as e:
                self._show_warning("Invalid Max Eccentricity", str(e))
                return False

        if name == "min_solidity":
            try:
                self.params.min_solidity = value
            except ValidationError as e:
                self._show_warning("Invalid Min Solidity", str(e))
                return False

        if name == "min_size":
            try:
                self.params.min_size = value
            except ValidationError as e:
                self._show_warning("Invalid Min Size", str(e))
                return False

        return True

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self.gui, title, message)

    def update_sample_image(self, direction: str) -> None:
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
        self.preview_image()

    # -------Second Column Functions-------------------------
    def initialize_algorithm_combo(self) -> None:
        # Initialize the algorithm combo box
        # And achieve all the available methods' names

        for algorithm, params in self.model.all_methods_n_params.items():
            print("initialize algorithm:", algorithm)
            self.algorithm_list.append(algorithm)

        self.gui.algorithm_combo.addItems(self.algorithm_list)
        self.update_model_algorithm(self.algorithm_list[0])

    def load_parameter_table_1(self, algorithm: str) -> None:
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
                    self.gui.param_sandbox1.setItem(
                        row, 1, QTableWidgetItem(str(value))
                    )
                    self.temp_param_dict[name] = value

                break

    def handle_algorithm_change(self, new_algorithm: str) -> None:
        # Update the algorithm in the model
        self.update_segment_parameters()

        # Reload the param table in the GUI
        self.load_parameter_table_1(new_algorithm)

        # Update params in the model
        self.update_model_algorithm(new_algorithm)

    def update_model_algorithm(self, algorithm: str) -> None:
        self.model.algorithm = algorithm

    def preview_processed_images(self) -> None:
        if_img, img_before_filter, img_after_filter = (
            self.model.preview_processed_image(self.current_index)
        )

        if if_img:
            self.update_label_before_filtering(img_before_filter)
            self.update_process_image_preview(img_after_filter)
        else:
            self._show_warning(
                "Image Not Found", "Image has not been fully processed yet."
            )

    def confirm_parameter_before_filtering(self) -> None:
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
        try:
            f = float(s)
            # Check if it has a fractional part
            print(s, "is float?")
            return not f.is_integer()
        except ValueError:
            print(s, "is not float")
            return False

    def convert_value(self, text: str) -> int | float | str:
        try:
            value = float(text)
            # Return an int if the number is integer
            return int(value) if value.is_integer() else value
        except ValueError:
            # Fallback if not a number
            return text

    def extract_parameters_from_table(self, table_widget) -> dict[str, int|float]: # type: ignore
        params = {}
        row_count = table_widget.rowCount()
        for row in range(row_count):
            name_item = table_widget.item(row, 0)
            value_item = table_widget.item(row, 1)
            if name_item and value_item:
                params[name_item.text()] = self.convert_value(value_item.text())
        return params  # type: ignore

    def update_segment_parameters(self) -> bool:
        # Update the params in the model
        # Extract parameters from the table
        params = self.extract_parameters_from_table(self.gui.param_sandbox1)

        # Update the model's dictionary for the selected algorithm
        for algorithm_name, params_in_dict in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for name, value in params.items():
                    print("Updating", name, "to", value)
                    params_in_dict[name] = value 

        return True

    def _process_step_1(self) -> None:
        step_1_img = self.model.step_1_main(self.current_index)
        self.update_label_before_filtering(step_1_img)

    def update_label_before_filtering(self, img: npt.NDArray[np.int_]) -> None:
        self.gui.label_before_filtering.axes.clear()
        self.gui.label_before_filtering.axes.imshow(img)
        self.gui.label_before_filtering.draw()

    # -------Third Column Functions: Filtering-------------------------
    def initialize_parameter_table_2(self) -> None:
        params = [
            ("max_eccentricity", self.temp_filter_param_dict["max_eccentricity"]),
            ("min_solidity", self.temp_filter_param_dict["min_solidity"]),
            ("min_size", self.temp_filter_param_dict["min_size"]),
        ]

        print("filter param dict---------------------:", self.temp_filter_param_dict)

        self.gui.param_sandbox2.setRowCount(len(params))

        for row, (name, value) in enumerate(params):
            self.gui.param_sandbox2.setItem(row, 0, QTableWidgetItem(name))
            self.gui.param_sandbox2.setItem(row, 1, QTableWidgetItem(str(value)))

    def confirm_parameter_for_filtering(self) -> None:
        for name, value in self.temp_filter_param_dict.items():
            if_valid = self.check_params(name, value)
            if not if_valid:
                return

        self.store_filter_params()
        self.pass_filter_params(self.temp_filter_param_dict)
        self._process_step_2()

    def store_filter_params(self) -> None:
        params = {}

        for row in range(self.gui.param_sandbox2.rowCount()):
            name_item = self.gui.param_sandbox2.item(row, 0)
            value_item = self.gui.param_sandbox2.item(row, 1)
            if name_item and value_item:
                params[name_item.text()] = float(value_item.text())

        self.temp_filter_param_dict["max_eccentricity"] = params.get(
            "max_eccentricity", 1.0
        )
        self.temp_filter_param_dict["min_solidity"] = params.get("min_solidity", 0.0)
        self.temp_filter_param_dict["min_size"] = params.get("min_size", 0)

    def _process_step_2(self) -> None:
        step_2_img = self.model.step_2_main(self.current_index)
        self.update_process_image_preview(step_2_img)

    # -------Third Column Functions: Manual Ellipse Adjustment-------------------------
    def ellipse_manual_adjustment(self) -> None:
        img = self.model.ellipse_manual_adjustment(self.current_index)
        self.update_process_image_preview(img)

    def update_process_image_preview(self, img: npt.NDArray[np.int_]) -> None:
        self.gui.processed_image_preview.axes.clear()
        self.gui.processed_image_preview.axes.imshow(img)
        self.gui.processed_image_preview.draw()

    # -------Third Column Functions: Batch Processing----------------------------
    def ask_if_batch(self) -> None:
        """Function to handle the batch processing of all images in the folder."""
        confirm_dialog = self.create_confirm_dialog()
        save_images_checkbox = self.create_save_images_checkbox(confirm_dialog)

        response = confirm_dialog.exec()

        if response == QMessageBox.StandardButton.Ok:
            self.batch_process_images()
        else:
            print("Batch processing canceled.")

    def create_confirm_dialog(self) -> QMessageBox:
        confirm_dialog = QMessageBox(self.gui)
        confirm_dialog.setWindowTitle("Batch Processing Confirmation")
        confirm_dialog.setText(
            "The parameters will be applied to all the images. Confirm to process."
        )
        confirm_dialog.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        return confirm_dialog

    def create_save_images_checkbox(self, dialog: QMessageBox) -> QCheckBox:
        save_images_checkbox = QCheckBox("Save processed images")
        save_images_checkbox.stateChanged.connect(
            lambda: self.update_if_save_processed_images(
                save_images_checkbox.isChecked()
            )
        )
        dialog.setCheckBox(save_images_checkbox)
        return save_images_checkbox

    def update_if_save_processed_images(self, state: bool) -> None: 
        self.if_save_processed_images = state

    def batch_process_images(self) -> None:
        for name, value in self.temp_param_dict.items():
            if_valid = self.check_params(name, value)
            if not if_valid:
                return

        for name, value in self.temp_filter_param_dict.items():
            if_valid = self.check_params(name, value)
            if not if_valid:
                return

        self.update_segment_parameters()
        self.pass_filter_params(self.temp_filter_param_dict)
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

        self.worker_thread = WorkerThread(
            self.model, self.if_save_processed_images, self.export_handler.save_path
        )

        self.worker_thread.update_progress.connect(self.update_progress_bar)
        self.worker_thread.processing_done.connect(self.on_processing_done)
        self.worker_thread.start()

    def show_progress_window(self, num_images: int) -> None:
        """Create and show a progress window with a loading bar."""
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
        """Update the progress bar value."""
        self.progress_bar.setValue(value)
        print("update progress bar:", value)

    def on_processing_done(self) -> None:
        """Handle the completion of image processing."""
        # Close the progress dialog
        self.progress_dialog.close()
        self.batch_processing_done.emit()

        # Switch to the final tab
        self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.results_tab))


class ResultsTabHandler:
    def __init__(self, params: Config) -> None:
        # self.gui = gui
        self.params = params
        self.save_path = params.save_path

        self.ellipses_properties: list[list[dict[str, float]]]
        self.export_handler: ExportSettingsHandler

    def load_gui(self, gui) -> None: # type: ignore
        self.gui = gui

    def load_ellipse_properties(self, properties: list[list[dict[str, float]]]) -> None:
        self.ellipses_properties = properties
        pass

    def generate_histogram(self) -> None:
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
        desc_text = (
            f"Results:\nd32 = {d32:.2f} mm\ndmean = {d_mean:.2f} mm\ndxy = {dxy:.2f} mm"
        )
        self.gui.descriptive_size_label.setText(desc_text)

        # Optionally add CDF
        if show_pdf or show_cdf:
            self.gui.histogram_canvas.axes2 = self.gui.histogram_canvas.axes.twinx()
            self.gui.histogram_canvas.axes2.set_ylabel("Probability [%]")

            if show_cdf:
                cdf = np.cumsum(counts) / np.sum(counts) * 100
                self.gui.histogram_canvas.axes2.plot(
                    bins[:-1], cdf, "r-", marker="o", label="CDF"
                )

            if show_pdf:
                pdf = counts / np.sum(counts) * 100
                self.gui.histogram_canvas.axes2.plot(
                    bins[:-1], pdf, "b-", marker="o", label="PDF"
                )

        if show_d32:
            self.gui.histogram_canvas.axes.axvline(
                x=d32, color="r", linestyle="-", label="d32"
            )

        if show_dmean:
            self.gui.histogram_canvas.axes.axvline(
                x=d_mean, color="g", linestyle="--", label="dmean"
            )

        if show_dxy:
            self.gui.histogram_canvas.axes.axvline(
                x=dxy, color="b", linestyle="--", label="dxy"
            )

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
                lines2, labels2 = (
                    self.gui.histogram_canvas.axes2.get_legend_handles_labels()
                )
                self.gui.histogram_canvas.axes.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc=legend_location_map.get(legend_position, "upper right"),
                )

        # Redraw the canvas
        self.gui.histogram_canvas.draw()

        return

    def calculate_descriptive_sizes(
        self, equivalent_diameters: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Calculate d32, d mean, and dxy based on the equivalent diameters."""
        dxy_x_power: int = int(self.gui.dxy_x_input.text())
        dxy_y_power: int = int(self.gui.dxy_y_input.text())
        d32: float = np.sum(equivalent_diameters**3) / np.sum(equivalent_diameters**2)

        # d32, Sauter diameter, should be calculated based on the area, and volume
        # diameter of a circle, which is unkown right now

        d_mean: float = float(np.mean(equivalent_diameters))
        dxy: float = np.sum(equivalent_diameters**dxy_x_power) / np.sum(
            equivalent_diameters**dxy_y_power
        )

        return d32, d_mean, dxy

    def get_equivalent_diameters_list(self) -> npt.NDArray[np.float64]:
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
            self._show_warning(
                "Folder Not Found", "Please select a valid folder in export settings."
            )
            return

        # Get the user-specified filenames
        graph_filename = self.gui.graph_filename_edit.text()
        csv_filename = self.gui.csv_filename_edit.text()

        if not graph_filename or not csv_filename:
            QMessageBox.warning(
                self,
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
            "major_axis_length" "minor_axis_length" "equivalent_diameter",
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

        self._show_warning(
            "Results Saved", f"Results have been saved successfully to {folder_path}."
        )
        return

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self.gui, title, message)


class MainHandler:
    def __init__(self) -> None:
        self.toml_file_path = Path("bubble_analyser/gui/config.toml")

        self.initialize_handlers()
        self.initialize_handlers_signals()

        self.initialize_gui()
        self.load_gui_for_handlers()

        self.load_export_settings()
        self.load_full_gui()

    def initialize_handlers(self) -> None:
        self.toml_handler = TomlFileHandler(self.toml_file_path)

        self.input_file_model = InputFilesModel()
        self.folder_tab_handler = FolderTabHandler(
            self.input_file_model, params=self.toml_handler.params
        )

        self.calibration_model = CalibrationModel()
        self.calibration_tab_handler = CalibrationTabHandler(
            self.calibration_model, params=self.toml_handler.params
        )

        self.image_processing_model = ImageProcessingModel(
            params=self.toml_handler.params
        )
        self.image_processing_tab_handler = ImageProcessingTabHandler(
            self.image_processing_model, params=self.toml_handler.params
        )

        self.results_tab_handler = ResultsTabHandler(params=self.toml_handler.params)

    def initialize_handlers_signals(self) -> None:
        self.image_processing_tab_handler.batch_processing_done.connect(
            self.start_generate_histogram
        )

    def initialize_gui(self) -> None:
        from bubble_analyser.gui import MainWindow

        self.app = QApplication(sys.argv)
        self.gui: MainWindow = MainWindow(self)

    def load_full_gui(self) -> None:
        self.gui.load_full_gui()
        self.gui.show()
        sys.exit(self.app.exec())

    def load_gui_for_handlers(self) -> None:
        self.folder_tab_handler.load_gui(self.gui)
        self.calibration_tab_handler.load_gui(self.gui)
        self.image_processing_tab_handler.load_gui(self.gui)
        self.results_tab_handler.load_gui(self.gui)

    def load_export_settings(self) -> None:
        self.export_handler = ExportSettingsHandler(
            parent=self.gui, params=self.toml_handler.params
        )
        self.image_processing_tab_handler.export_handler = self.export_handler
        self.results_tab_handler.export_handler = self.export_handler

    def menubar_open_export_settings_dialog(self) -> None:
        self.export_handler.exec()

    def menubar_ask_if_restart(self) -> None:
        restart = QMessageBox.question(
            self.gui,
            "Restart",
            "Do you want to start a new mission? \n Current unsaved progress will be lost",
            QMessageBox.Yes | QMessageBox.No,
        )
        if restart == QMessageBox.Yes:
            self.restart()

    def tab1_select_folder(self) -> None:
        self.folder_tab_handler.select_folder()

    def tab1_confirm_folder_selection(self) -> None:
        self.folder_tab_handler.confirm_folder_selection()
        # load image path list to Processing Hanlder
        self.image_processing_model.confirm_folder_selection(
            self.input_file_model.image_list_full_path_in_path
        )

    def tab2_get_px2mm_ratio(self) -> None:
        self.calibration_tab_handler.get_px2mm_ratio()

    def tab2_select_px_mm_image(self) -> None:
        self.calibration_tab_handler.select_px_mm_image()

    def tab2_select_bg_corr_image(self) -> None:
        self.calibration_tab_handler.select_bg_corr_image()

    def tab2_confirm_calibration(self) -> None:
        self.calibration_tab_handler.confirm_calibration()
        self.image_processing_model.update_px2mm(self.calibration_model.px2mm)

        if self.calibration_model.if_bknd:
            self.image_processing_model.if_bknd = self.calibration_model.if_bknd
            self.image_processing_model.get_bknd_img_path(
                self.calibration_model.bknd_img_path
            )

    def tab3_load_parameter_table_1(self, algorithm: str) -> None:
        self.image_processing_tab_handler.load_parameter_table_1(algorithm)

    def tab3_initialize_parameter_table_2(self) -> None:
        self.image_processing_tab_handler.initialize_parameter_table_2()

    def tab3_update_sample_image(self, status: str) -> None:
        self.image_processing_tab_handler.update_sample_image(status)

    def tab3_preview_processed_images(self) -> None:
        self.image_processing_tab_handler.preview_processed_images()

    def tab3_handle_algorithm_change(self, algorithm: str) -> None:
        self.image_processing_tab_handler.handle_algorithm_change(algorithm)

    def tab3_confirm_parameter_before_filtering(self) -> None:
        self.image_processing_tab_handler.confirm_parameter_before_filtering()

    def tab3_confirm_parameter_for_filtering(self) -> None:
        self.image_processing_tab_handler.confirm_parameter_for_filtering()

    def tab3_ellipse_manual_adjustment(self) -> None:
        self.image_processing_tab_handler.ellipse_manual_adjustment()

    def tab3_ask_if_batch(self) -> None:
        self.image_processing_tab_handler.ask_if_batch()

    def start_generate_histogram(self) -> None:
        self.results_tab_handler.load_ellipse_properties(
            self.image_processing_model.ellipses_properties 
        )
        self.results_tab_handler.generate_histogram()

    def save_results(self) -> None:
        self.results_tab_handler.save_results()

    def restart(self) -> None:
        # self.toml_handler = None
        # self.input_file_model = None
        # self.folder_tab_handler = None
        # self.calibration_model = None
        # self.calibration_tab_handler = None
        # self.image_processing_model = None
        # self.image_processing_tab_handler = None
        # self.results_tab_handler = None
        # self.export_handler = None

        # Close the existing QApplication instance
        # self.app.quit()
        # self.app = None

        # QCoreApplication.quit()
        # QTimer.singleShot(0, self.reinitialize_main_handler)

        # Start a new detached process running the same Python executable with the same arguments
        QProcess.startDetached(sys.executable, sys.argv)
        # Quit the current application
        QApplication.quit()

    # def reinitialize_main_handler(self) -> None:
    #     self.__init__()


if __name__ == "__main__":
    main_handler = MainHandler()
