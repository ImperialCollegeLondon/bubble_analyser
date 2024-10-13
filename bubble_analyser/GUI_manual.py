"""GUI Manual Module: A graphical user interface (GUI) for the Bubble Analyser.

This module provides a graphical user interface (GUI) for the Bubble Analyser
application. It contains classes and functions for creating and managing the GUI,
including the main window, image processing, and data visualization.

Author: Yiyang Guan
Date: 06-Oct-2024

Classes:
    MplCanvas: A class for creating a Matplotlib figure within a PySide6 application.
    MainWindow: The main window of the GUI application.

"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import toml as tomllib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import typing as npt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage import (
    morphology,
)

from .calculate_px2mm import calculate_px2mm
from .config import Config
from .default import run_watershed_segmentation
from .image_preprocess import image_preprocess
from .morphological_process import morphological_process
from .threshold import threshold, threshold_without_background


class MplCanvas(FigureCanvas):
    """A class for creating a Matplotlib figure within a PySide6 application.

    Attributes:
        fig: The Matplotlib figure.
        axes: The axes of the figure.
    """

    def __init__(
        self, parent: QMainWindow, width: float = 5, height: float = 4, dpi: float = 100
    ) -> None:
        """The constructor for MplCanvas.

        Parameters:
            parent: The parent widget.
            width: The width of the figure in inches.
            height: The height of the figure in inches.
            dpi: The dots per inch of the figure.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    """The main application window for the Bubble Analyser GUI.

    This class is responsible for loading the configuration parameters, setting up the
    window title and geometry, and creating the main widgets, including the folder,
    calibration, image processing, and results tabs.

    Attributes:
        params (Config): The configuration parameters loaded from the TOML file.
        img_resample_factor (float): The image resampling factor.
        threshold_value (float): The threshold value for image processing.
        element_size (int): The size of the morphological element.
        connectivity (int): The connectivity for image processing.
        max_eccentricity (float): The maximum eccentricity for feature detection.
        min_solidity (float): The minimum solidity for feature detection.

    Methods:
        load_toml: Loads the configuration parameters from the TOML file.
        setup_folder_tab: Sets up the folder tab widget.
        setup_calibration_tab: Sets up the calibration tab widget.
        setup_image_processing_tab: Sets up the image processing tab widget.
        setup_results_tab: Sets up the results tab widget.
    """

    def __init__(self) -> None:
        """The constructor for the main window.

        Loads the configuration parameters from the TOML file, sets the window title and
        geometry, and creates the main widgets, including the folder, calibration, image
        processing and results tabs.
        """
        super().__init__()

        self.params = self.load_toml("./bubble_analyser/config.toml")
        self.img_resample_factor = self.params.resample
        self.threshold_value = self.params.threshold_value
        self.element_size = self.params.Morphological_element_size
        self.connectivity = self.params.Connectivity
        self.max_eccentricity = self.params.Max_Eccentricity
        self.min_solidity = self.params.Min_Solidity
        self.min_circularity = self.params.Min_Circularity
        self.min_size = self.params.min_size

        self.bknd_img_exist = False
        self.calibration_confirmed = False

        self.setWindowTitle("Bubble Analyser")
        self.setGeometry(100, 100, 1200, 800)

        # Create a Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add Folder Tab
        self.folder_tab = QWidget()
        self.tabs.addTab(self.folder_tab, "Folder")
        self.sample_images_confirmed = False
        self.setup_folder_tab()

        # Add Calibration Tab
        self.calibration_tab = QWidget()
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.bg_image_confirmed = False
        self.px_res_confirmed = False
        self.setup_calibration_tab()

        # Add Image Processing Tab
        self.image_processing_tab = QWidget()
        self.tabs.addTab(self.image_processing_tab, "Bubble detection and filtering")
        self.setup_image_processing_tab()

        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Results")
        self.setup_results_tab()

    def load_toml(self, file_path: str) -> Config:
        """Load configuration parameters from a TOML file.

        This function reads the TOML configuration file from the specified path and
        loads its contents into a dictionary.

        Args:
            file_path: The file path of the TOML configuration file.

        Returns:
            A dictionary containing the configuration parameters from the TOML file.
        """
        toml_data = tomllib.load(file_path)

        return Config(**toml_data)

    def setup_folder_tab(self) -> None:
        """Set up the folder tab.

        This function sets up the folder tab, which contains the following components:
        1. A text box for user to input the folder path.
        2. A button to select the folder.
        3. A button to confirm the folder selection.
        4. A list of images in the selected folder.
        5. An image preview section to show the selected image.

        When the user selects an image from the list, the image will be previewed in
        the image preview section.
        """
        layout = QVBoxLayout(self.folder_tab)

        # Top Part: Folder Selection
        top_frame = QFrame()
        top_layout = QHBoxLayout(top_frame)
        self.folder_path_edit = QLineEdit()
        select_folder_button = QPushButton("Select Folder")
        confirm_folder_button = QPushButton("Confirm Folder")
        select_folder_button.clicked.connect(self.select_folder)
        confirm_folder_button.clicked.connect(self.confirm_folder_selection)
        top_layout.addWidget(select_folder_button)
        top_layout.addWidget(self.folder_path_edit)
        top_layout.addWidget(confirm_folder_button)

        # Bottom Left: List of images in folder
        bottom_left_frame = QFrame()
        bottom_left_layout = QVBoxLayout(bottom_left_frame)
        self.image_list = QListWidget()
        self.image_list.clicked.connect(self.preview_image)
        bottom_left_layout.addWidget(self.image_list)

        # Bottom Right: Image Preview
        bottom_right_frame = QFrame()
        bottom_right_layout = QVBoxLayout(bottom_right_frame)
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setFixedSize(
            600, 600
        )  # Set a fixed size for the image preview
        bottom_right_layout.addWidget(self.image_preview)

        # Split the bottom part into two sections
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.addWidget(bottom_left_frame)
        bottom_layout.addWidget(bottom_right_frame)

        # Add top and bottom frames to the main layout
        layout.addWidget(top_frame, 1)
        layout.addWidget(bottom_frame, 6)

    def select_folder(self) -> None:
        """Select a folder.

        Open a folder selection dialog and update the folder path edit
        and image list if a valid folder is selected. If the sample images
        have already been confirmed, display a warning message and do nothing.
        """
        if self.sample_images_confirmed:
            QMessageBox.warning(
                self,
                "Selection Locked",
                "You have already confirmed the folder selection.",
            )
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.populate_image_list(folder_path)

    def confirm_folder_selection(self) -> None:
        """Confirm the selection of folder.

        Confirm the folder selection and lock the folder path edit. If the
        selection has already been confirmed, display a warning message and do
        nothing. Otherwise, set the folder path edit to read-only, load the
        images to process from the selected folder, and switch to the next tab.
        """
        if not self.sample_images_confirmed:
            self.sample_images_confirmed = True
            # Lock the folder path edit and confirm the selection
            self.folder_path_edit.setReadOnly(True)
            self.load_images_to_process()
            self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1)

        else:
            QMessageBox.warning(
                self,
                "Selection Locked",
                "You have already confirmed the folder selection.",
            )

    def populate_image_list(self, folder_path: str) -> None:
        """Popultate the image list with the names of images in the given folder path.

        Populate the image list with the names of images in the given folder path,
        and store the full paths to the images in the image_list_full_path list.

        This function clears the image list, and then iterates over the files in the
        given folder path. If a file has an extension matching a common image
        format (e.g., .png, .jpg, .jpeg, .bmp, .tiff), it adds the file name to the
        image list and the full path to the image_list_full_path list.

        The purpose of this function is to populate the image list in the GUI with
        the names of images in the selected folder, so that the user can select
        specific images to process. The full paths to the selected images are stored
        in the image_list_full_path list, and are used later to load the images when
        the user clicks the "Next" button.
        """
        self.image_list.clear()
        self.image_list_full_path: list[str] = []

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self.image_list.addItem(file_name)
                self.image_list_full_path.append(os.path.join(folder_path, file_name))

        print("self_image_list_full_path", self.image_list_full_path)

    def preview_image(self) -> None:
        """Preview the currently selected image.

        Preview the currently selected image in the GUI. This function is
        called when the user selects an image from the image list. It gets the
        currently selected image, loads it as a QPixmap, and sets it to the
        image preview label on the GUI. The image is scaled to fit the size of
        the label while keeping the aspect ratio.
        """
        self.selected_image = self.image_list.currentItem().text()
        folder_path = self.folder_path_edit.text()
        image_path = os.path.join(folder_path, self.selected_image)
        pixmap = QPixmap(image_path)

        self.image_preview.setPixmap(
            pixmap.scaled(self.image_preview.size(), Qt.KeepAspectRatio)
        )

        self.sample_image_preview.setPixmap(
            pixmap.scaled(self.sample_image_preview.size(), Qt.KeepAspectRatio)
        )

    def load_images_to_process(self) -> None:
        """Load images to process.

        Populate the image list with the names of images in the folder path
        set in the folder path edit, and store the full paths to the images in
        the image_list_full_path list. This function is called after the user
        confirms the folder selection. The purpose of this function is to
        populate the image list in the GUI with the names of images in the
        selected folder, so that the user can select specific images to
        process. The full paths to the selected images are stored in the
        image_list_full_path list, and are used later to load the images when
        the user clicks the "Next" button.
        """
        folder_path = self.folder_path_edit.text()
        if os.path.exists(folder_path):
            self.populate_image_list(folder_path)

    def setup_calibration_tab(self) -> None:
        """Set up the calibration tab.

        Set up the calibration tab, which contains the following components:

        1. A text box for user to input the name of the image for pixel
           resolution calibration.
        2. A button to select the image for pixel resolution calibration.
        3. A label to preview the selected image.
        4. A text box for user to input the name of the background image.
        5. A button to select the background image.
        6. A label to preview the selected background image.
        7. A button to confirm the calibration and background image.

        When the user selects an image from the list, the image will be
        previewed in the image preview section. When the user clicks the
        "Confirm" button, the image will be processed and the pixel-to-mm
        ratio will be calculated and stored in the "px_mm" attribute of the
        MainWindow object. The background image will be stored in the
        "bknd_img" attribute of the MainWindow object. The tab will then be
        switched to the next tab.
        """
        layout = QGridLayout(self.calibration_tab)

        # Create top frame
        top_frame = QFrame()
        top_frame_layout = QHBoxLayout()
        top_frame.setLayout(top_frame_layout)

        # Pixel Resolution Calibration
        px_res_frame = QFrame()
        px_res_layout = QVBoxLayout(px_res_frame)
        self.px_res_image_name = QLineEdit()
        self.px_res_image_name.setText("Choose your ruler image from local")
        px_res_confirm_button = QPushButton("Confirm")
        px_res_confirm_button.clicked.connect(self.process_calibration_image)
        px_res_select_button = QPushButton(
            "Select other image for resolution calibration"
        )
        px_res_select_button.clicked.connect(self.select_px_res_image)
        self.px_res_image_preview = QLabel()
        self.px_res_image_preview.setAlignment(Qt.AlignCenter)
        self.px_res_image_preview.setFixedSize(400, 300)

        px_res_layout.addWidget(QLabel("Step 1: Pixel resolution calibration"))
        px_res_layout.addWidget(QLabel("Image name"))
        px_res_layout.addWidget(self.px_res_image_name)
        px_res_layout.addWidget(px_res_confirm_button)
        px_res_layout.addWidget(px_res_select_button)
        px_res_layout.addWidget(self.px_res_image_preview)
        px_res_layout.addStretch()

        # Background Correction Image
        bg_corr_frame = QFrame()
        bg_corr_layout = QVBoxLayout(bg_corr_frame)
        self.bg_corr_image_name = QLineEdit()
        self.bg_corr_image_name.setText("Choose your background image from local")
        # bg_corr_confirm_button = QPushButton("Confirm")
        # bg_corr_confirm_button.clicked.connect(self.confirm_bg_corr_image)
        bg_corr_select_button = QPushButton(
            "Select other image for background correction"
        )
        bg_corr_select_button.clicked.connect(self.select_bg_corr_image)
        self.bg_corr_image_preview = QLabel()
        self.bg_corr_image_preview.setAlignment(Qt.AlignCenter)
        self.bg_corr_image_preview.setFixedSize(400, 300)

        bg_corr_layout.addWidget(QLabel("Step 2: Background correction image"))
        bg_corr_layout.addWidget(QLabel("Image name"))
        bg_corr_layout.addWidget(self.bg_corr_image_name)
        # bg_corr_layout.addWidget(bg_corr_confirm_button)
        bg_corr_layout.addWidget(bg_corr_select_button)
        bg_corr_layout.addWidget(self.bg_corr_image_preview)
        bg_corr_layout.addStretch()

        # Adding Pixel Resolution and Background Correction frames to top layout
        top_frame_layout.addWidget(px_res_frame)
        top_frame_layout.addWidget(bg_corr_frame)

        # Create bottom frame for manual calibration input
        bottom_frame = QFrame()
        bottom_frame_layout = QVBoxLayout()
        bottom_frame.setLayout(bottom_frame_layout)

        manualcalibration_frame = QFrame()
        manualcalibration_layout = QHBoxLayout(manualcalibration_frame)
        manualcalibration_layout.addWidget(QLabel("or:"))
        self.manual_px_mm_input = QLineEdit()
        self.manual_px_mm_input.setPlaceholderText("Calibrate manually")
        manualcalibration_layout.addWidget(self.manual_px_mm_input)
        manualcalibration_layout.addWidget(QLabel("px/mm"))

        confirm_px_mm_button = QPushButton("Confirm calibration and background image")
        confirm_px_mm_button.clicked.connect(
            self.confirm_calibration
        )  # Connect confirm button to the handler

        bottom_frame_layout.addWidget(manualcalibration_frame)
        bottom_frame_layout.addWidget(confirm_px_mm_button)

        # Add frames to main layout
        layout.addWidget(top_frame, 0, 0, 1, 2)
        layout.addWidget(bottom_frame, 1, 0, 1, 2)

    def confirm_bg_corr_image(self) -> None:
        """Confirm the background correction image and lock the input."""
        if not self.calibration_confirmed:
            self.bg_image_confirmed = True
            self.bg_corr_image_name.setReadOnly(True)  # Lock the input field
        else:
            QMessageBox.warning(
                self,
                "Selection Locked",
                "You have already confirmed the background selection.",
            )

    def select_px_res_image(self) -> None:
        """Open a file dialog to select an image for pixel resolution calibration.

        This function will lock if the calibration has already been confirmed.

        If a valid image path is selected, the image name will be displayed in the
        corresponding text box and the image will be previewed in the image preview
        section. The image is scaled to fit the size of the label while keeping the
        aspect ratio.
        """
        if self.calibration_confirmed:
            QMessageBox.warning(
                self,
                "Selection Locked",
                "You have already confirmed the pixel resolution.",
            )
            return
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image for Resolution Calibration",
            "",
            "Image Files (*.png *.jpg *.bmp)",
        )

        if image_path:
            self.px_res_image_name.setText(image_path)
            pixmap = QPixmap(image_path)
            self.px_res_image_preview.setPixmap(
                pixmap.scaled(self.px_res_image_preview.size(), Qt.KeepAspectRatio)
            )

    def process_calibration_image(self) -> None:
        """Process the selected image for pixel resolution calibration.

        If a valid image path is selected, calculate the pixel-to-mm ratio and
        display it in the manual calibration text box. If the image path is not
        valid, display a status bar message. The function will lock if the
        calibration has already been confirmed.

        Args:
            None.

        Returns:
            None.
        """
        if self.calibration_confirmed:
            QMessageBox.warning(
                self,
                "Selection Locked",
                "You have already confirmed the pixel resolution.",
            )
            return

        image_path: Path = Path(self.px_res_image_name.text())
        if image_path and os.path.exists(image_path):
            __, mm2px = calculate_px2mm(
                image_path, img_resample=0.5
            )  # Use the stored full path
            self.manual_px_mm_input.setText(f"{mm2px:.3f}")
        else:
            self.statusBar().showMessage(
                "Image file does not exist or not selected.", 5000
            )

    def select_bg_corr_image(self) -> None:
        """Open a file dialog to select an image for background correction.

        If the background correction has already been confirmed, display a warning
        message and do nothing. Otherwise, open a file dialog to select an image.
        If a valid image path is selected, display the image in the background
        correction image preview section, and set the path to the background
        correction image name text box. The background image existence flag is
        also set to True.

        Args:
            None.

        Returns:
            None.
        """
        if self.calibration_confirmed:
            QMessageBox.warning(
                self,
                "Selection Locked",
                "You have already confirmed the background selection.",
            )
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image for Background Correction",
            "",
            "Image Files (*.png *.jpg *.bmp)",
        )
        if image_path:
            self.bg_corr_image_name.setText(image_path)
            pixmap = QPixmap(image_path)
            self.bg_corr_image_preview.setPixmap(
                pixmap.scaled(self.bg_corr_image_preview.size(), Qt.KeepAspectRatio)
            )
            self.bknd_img_exist = True

    def confirm_calibration(self) -> None:
        """Confirm the manual calibration and lock the input."""
        if not self.calibration_confirmed:
            self.calibration_confirmed = True
            self.px2mm = float(
                self.manual_px_mm_input.text()
            )  # Store the pixel to mm ratio
            self.mm2px = 1 / self.px2mm
            self.manual_px_mm_input.setReadOnly(True)  # Lock the input field
            self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1)
        else:
            QMessageBox.warning(
                self,
                "Selection Locked",
                "You have already confirmed the calibration process.",
            )

    def setup_image_processing_tab(self) -> None:
        """Set up the image processing tab, which contains the following components.

        Set up the image processing tab, which contains the following components:
        1. A box to select the image processing algorithm.
        2. A table to display and edit the processing parameters.
        3. A button to confirm the parameter settings and preview a sample image.
        4. A button to batch process the sample images.
        5. A section to display the sample image preview.
        6. A section to display the processed image preview, including before and after
        filtering.

        When the user selects an algorithm, the parameters will be loaded and displayed
        in the table. When the user confirms the parameter settings and previews a
        sample image, the sample image will be processed using the selected algorithm
        and displayed in the preview section.
        When the user clicks the "Batch process images" button, the algorithm will be
        applied to all the images in the selected folder and the results will be saved
        in a new folder.
        """
        layout = QGridLayout(self.image_processing_tab)

        # Left Column: Algorithm selection and processing options
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)

        # Algorithm selection box
        algorithm_selection = QComboBox()
        algorithm_selection.addItems(["Default algorithm"])
        left_layout.addWidget(QLabel("Step 1: Select image processing algorithm"))
        left_layout.addWidget(algorithm_selection)

        # Add processing options to layout
        left_layout.addWidget(QLabel("Step 2:"))
        left_layout.addWidget(QLabel("Image processing sandbox"))
        # left_layout.addWidget(default_params_radio)
        # left_layout.addWidget(custom_params_radio)

        # Parameter table
        self.param_table = QTableWidget(8, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])

        # Populate the table with initial values
        self.param_table.setItem(0, 0, QTableWidgetItem("img_resample_factor"))
        self.param_table.setItem(0, 1, QTableWidgetItem(str(self.img_resample_factor)))

        self.param_table.setItem(1, 0, QTableWidgetItem("threshold_value"))
        self.param_table.setItem(1, 1, QTableWidgetItem(str(self.threshold_value)))

        self.param_table.setItem(2, 0, QTableWidgetItem("element_size"))
        self.param_table.setItem(2, 1, QTableWidgetItem(str(self.element_size)))

        self.param_table.setItem(3, 0, QTableWidgetItem("connectivity"))
        self.param_table.setItem(3, 1, QTableWidgetItem(str(self.connectivity)))

        self.param_table.setItem(4, 0, QTableWidgetItem("max_eccentricity"))
        self.param_table.setItem(4, 1, QTableWidgetItem(str(self.max_eccentricity)))

        self.param_table.setItem(5, 0, QTableWidgetItem("min_circularity"))
        self.param_table.setItem(5, 1, QTableWidgetItem(str(self.min_circularity)))

        self.param_table.setItem(6, 0, QTableWidgetItem("min_solidity"))
        self.param_table.setItem(6, 1, QTableWidgetItem(str(self.min_solidity)))

        self.param_table.setItem(7, 0, QTableWidgetItem("min_size"))
        self.param_table.setItem(7, 1, QTableWidgetItem(str(self.min_size)))

        left_layout.addWidget(self.param_table)

        # Add Confirm Parameter Button
        preview_button = QPushButton("Confirm parameter and preview")
        preview_button.clicked.connect(self.confirm_parameter_and_preview)
        left_layout.addWidget(preview_button)

        # Add Processing Button
        process_button = QPushButton("Batch process images")
        process_button.clicked.connect(self.ask_if_batch)
        left_layout.addWidget(process_button)

        # Middle Column: Sample Image Preview
        middle_frame = QFrame()
        middle_layout = QVBoxLayout(middle_frame)
        self.sample_image_preview = QLabel("Sample image preview")
        self.sample_image_preview.setAlignment(Qt.AlignCenter)
        self.sample_image_preview.setFixedSize(400, 300)

        prev_button = QPushButton("< Prev. Img")
        next_button = QPushButton("Next Img >")
        prev_button.clicked.connect(lambda: self.update_sample_image("prev"))
        next_button.clicked.connect(lambda: self.update_sample_image("next"))

        middle_layout.addWidget(self.sample_image_preview)
        middle_button_layout = QHBoxLayout()
        middle_button_layout.addWidget(prev_button)
        middle_button_layout.addWidget(next_button)
        middle_layout.addLayout(middle_button_layout)

        # Right Column: Processed Image Preview
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)

        # Original Image Preview
        self.label_before_filtering = MplCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(QLabel("Processed Image_Before Filtering"))
        right_layout.addWidget(self.label_before_filtering)

        self.processed_image_preview = MplCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(QLabel("Processed Image_After Filtering"))
        right_layout.addWidget(self.processed_image_preview)

        # Add left, middle, and right frames to the main layout
        layout.addWidget(left_frame, 0, 0)
        layout.addWidget(middle_frame, 0, 1)
        layout.addWidget(right_frame, 0, 2)

    def confirm_parameter_and_preview(self) -> None:
        """Confirm the parameter settings and preview a sample image.

        This function is connected to the "Confirm parameter and preview" button
        in the image processing tab. When the button is clicked, the function
        checks the validity of the parameters, processes the selected image with
        the chosen algorithm, and displays the processed image on the right side
        of the tab.

        The function first checks the validity of the parameters, then processes
        the selected image with the chosen algorithm and displays the processed
        image on the right side of the tab. The processed image is displayed
        before and after filtering.

        :return: None
        """
        self.check_parameters()  # Check the validity of the parameters

        # Processing the image and displaying it on the right side
        selected_image = self.selected_image
        folder_path = self.folder_path_edit.text()
        image_path = os.path.join(folder_path, selected_image)

        imgThreshold, imgRGB = self.load_image_for_processing(image_path)
        preview_processed_image, labels_before_filtering, _, _ = self.run_processing(
            imgThreshold,
            imgRGB,
            self.mm2px,
            self.threshold_value,
            self.element_size,
            self.connectivity,
            self.max_eccentricity,
            self.min_solidity,
            self.min_circularity,
            self.min_size,
        )

        self.processed_image_preview.axes.clear()
        self.processed_image_preview.axes.imshow(preview_processed_image)
        self.processed_image_preview.draw()

        self.label_before_filtering.axes.clear()
        self.label_before_filtering.axes.imshow(labels_before_filtering)
        self.label_before_filtering.draw()

    def ask_if_batch(self) -> None:
        """Function to handle the batch processing of all images in the folder."""
        # Confirm dialog
        confirm_dialog = QMessageBox()
        confirm_dialog.setWindowTitle("Batch Processing Confirmation")
        confirm_dialog.setText(
            "The parameters will be applied to all the images. Confirm to process."
        )
        confirm_dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        response = confirm_dialog.exec()

        if response == QMessageBox.Ok:
            self.batch_process_images()
        else:
            print("Batch processing canceled.")

    def batch_process_images(self) -> None:
        """Function to handle the batch processing of all images in the folder.

        The parameters set by the user will be applied to all the images in the folder.
        The function processes each image one by one and stores the properties of all
        images in the `all_properties` list.

        :return: None
        """
        self.check_parameters()

        self.all_properties: list[
            dict[str, float]
        ] = []  # To store properties of all images

        for image_path in self.image_list_full_path:
            if image_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                print("Current processing image:", image_path)

                imgThreshold, imgRGB = self.load_image_for_processing(image_path)
                _, _, circle_properties, _ = self.run_processing(
                    imgThreshold,
                    imgRGB,
                    self.mm2px,
                    self.threshold_value,
                    self.element_size,
                    self.connectivity,
                    self.max_eccentricity,
                    self.min_solidity,
                    self.min_circularity,
                    self.min_size,
                )
                for properties in circle_properties:
                    self.all_properties.append(properties)
                # self.all_properties.append(circle_properties)
                print("Circle properties for this image:", circle_properties)
            print(
                "Batch processing completed. Circle properties for all images:",
                self.all_properties,
            )

    def check_parameters(self) -> None:
        """Check if parameters are valid numbers and calibration is confirmed.

        This function checks if the user has confirmed the calibration and if the
        parameters in the table are valid numbers. If not, it displays a warning
        message and returns without performing any action.

        Returns:
            None
        """
        if not self.calibration_confirmed:
            QMessageBox.warning(
                self,
                "Process Locked",
                "You have not yet confirmed the pixel resolution.",
            )
            return

        try:
            self.img_resample_factor = float(self.param_table.item(0, 1).text())
            self.threshold_value = float(self.param_table.item(1, 1).text())
            self.element_size = int(self.param_table.item(2, 1).text())
            self.connectivity = int(self.param_table.item(3, 1).text())
            self.max_eccentricity = float(self.param_table.item(4, 1).text())
            self.min_circularity = float(self.param_table.item(5, 1).text())
            self.min_solidity = float(self.param_table.item(6, 1).text())
            self.min_size = float(self.param_table.item(7, 1).text())

        except (ValueError, TypeError):
            QMessageBox.warning(
                self, "Invalid Input", "Please ensure all parameters are valid numbers."
            )
            return

    def load_image_for_processing(
        self, image_path: str
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Load and return the image, possibly applying some processing.

        This function loads an image using image_preprocess, applies background
        subtraction and thresholding, and morphological processing using a disk
        element of size Morphological_element_size. (If no background image is provided,
        the function applies thresholding without background subtraction.)

        Args:
            image_path (str): The path to the image to be processed.

        Returns:
            tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: A tuple of two arrays,
                where the first being the processed image and the second being the
                original image in RGB format.
        """
        target_image_path: Path = Path(image_path)
        target_img, imgRGB = image_preprocess(
            target_image_path, self.img_resample_factor
        )
        if self.bknd_img_exist:
            bg_img_path: Path = Path(self.bg_corr_image_name.text())
            self.bknd_img, _ = image_preprocess(bg_img_path, self.img_resample_factor)

            imgThreshold = threshold(target_img, self.bknd_img, self.threshold_value)
        else:
            imgThreshold = threshold_without_background(
                target_img, self.threshold_value
            )

        element_size = morphology.disk(self.params.Morphological_element_size)
        imgThreshold = morphological_process(imgThreshold, element_size)

        return imgThreshold, imgRGB

    def run_processing(
        self,
        imgThreshold: npt.NDArray[np.int_],
        imgRGB: npt.NDArray[np.int_],
        mm2px: float,
        threshold_value: float,
        element_size: int,
        connectivity: int,
        max_eccentricity: float,
        min_solidity: float,
        min_circularity: float,
        min_size: float,
    ) -> tuple[
        npt.NDArray[np.int_],
        npt.NDArray[np.int_],
        list[dict[str, float]],
        npt.NDArray[np.int_],
    ]:
        """Run the image processing algorithm on the preprocessed image.

        This function calls run_watershed_segmentation with the given parameters and
        returns the processed image, the labeled image before filtering, the properties
        of the detected circular features, and the labeled image after filtering.

        Parameters:
            imgThreshold (npt.NDArray[np.int_]): The preprocessed image after
            thresholding.
            imgRGB (npt.NDArray[np.int_]): The original image in RGB format.
            mm2px (float): The conversion factor from millimeters to pixels.
            threshold_value (float): The threshold value for background subtraction.
            element_size (int): The size of the morphological element for binary
            operations.
            connectivity (int): The connectivity of the morphological operations.
            max_eccentricity (float): The maximum eccentricity threshold for filtering.
            min_solidity (float): The minimum solidity threshold for filtering.
            min_circularity (float): The minimum circularity threshold for filtering.
            min_size (float): The minimum size threshold for filtering in pixels.

        Returns:
            tuple[npt.NDArray[np.int_],
                npt.NDArray[np.int_],
                list[dict[str, float]],
                npt.NDArray[np.int_]]: A tuple of four arrays, the first being the
                processed image, the second being the labeled image before filtering,
                the third being the properties of the detected circular features, and
                the fourth being the labeled image after filtering.
        """
        print("Threshold_value:", threshold_value)
        print("element_size:", element_size)
        print("connectivity:", connectivity)
        print("mm2px:", mm2px)
        print("max ecccentricity from GUI:", max_eccentricity)

        preview_processed_image, labels_before_filtering, circle_properties, labels = (
            run_watershed_segmentation(
                imgThreshold,
                imgRGB,
                mm2px,
                threshold_value,
                element_size,
                connectivity,
                max_eccentricity,
                min_solidity,
                min_circularity,
                min_size,
            )
        )
        return (
            preview_processed_image,
            labels_before_filtering,
            circle_properties,
            labels,
        )

    def update_sample_image(self, direction: str) -> None:
        """Update the sample image preview based on user navigation (prev/next).

        This function updates the sample image preview by changing the currently
        selected image in the image list. The direction parameter determines
        whether the user is navigating to the previous or next image. The
        function then calls the preview_image method to update the image preview.
        """
        current_row = self.image_list.currentRow()
        if direction == "prev":
            if current_row > 0:
                self.image_list.setCurrentRow(current_row - 1)
        elif direction == "next":
            if current_row < self.image_list.count() - 1:
                self.image_list.setCurrentRow(current_row + 1)
        self.preview_image()

    def setup_results_tab(self) -> None:
        """Set up the results tab.

        This function sets up the results tab by creating the following widgets:
        1. A graph canvas for displaying the histogram.
        2. Controls for histogram options, including the type of histogram to
        generate (by number or volume), checkboxes for PDF and CDF, the number
        of bins, and the x-axis limits.
        3. A legend position and orientation dropdown.
        4. A descriptive size options section, which includes checkboxes for
        displaying d32, dmean, and dxy, as well as input boxes for x and y
        values for dxy.
        5. A save button that saves the graph and CSV data to a user-selected
        folder.

        This function creates the layout of the results tab, including the graph
        canvas, controls, and save button. The controls include the histogram type,
        PDF/CDF checkboxes, number of bins, x-axis limits, legend position and
        orientation dropdown, and descriptive size options. The save button is
        connected to the save_results slot, which saves the graph and CSV data to
        the user-selected folder.
        """
        layout = QGridLayout(self.results_tab)

        # Create canvas for displaying the graph
        self.histogram_canvas = MplCanvas(self, width=8, height=8, dpi=100)

        # Controls for histogram options
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)

        # Histogram type
        histogram_by_label = QLabel("Histogram by:")
        self.histogram_by = QComboBox()
        self.histogram_by.addItems(["Number", "Volume"])

        # PDF/CDF Checkboxes
        self.pdf_checkbox = QCheckBox("PDF")
        self.cdf_checkbox = QCheckBox("CDF")

        # Number of bins
        bins_label = QLabel("Number of bins:")
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setValue(15)
        self.bins_spinbox.setRange(1, 100)

        # X-axis limits
        x_axis_limits_label = QLabel("X-axis limits:")
        self.min_x_axis_input = QLineEdit("0.0")
        self.max_x_axis_input = QLineEdit("5.0")

        legend_label = QLabel("Legend settings:")
        legend_frame = QFrame()
        legend_layout = QGridLayout(legend_frame)

        # Legend position dropdown
        legend_layout.addWidget(QLabel("Position:"), 0, 0)
        self.legend_position_combobox = QComboBox()
        self.legend_position_combobox.addItems(
            ["North East", "North West", "South East", "South West"]
        )
        legend_layout.addWidget(self.legend_position_combobox, 0, 1)

        # Legend orientation dropdown
        legend_layout.addWidget(QLabel("Orientation:"), 1, 0)
        self.legend_orientation_combobox = QComboBox()
        self.legend_orientation_combobox.addItems(["Vertical", "Horizontal"])
        legend_layout.addWidget(self.legend_orientation_combobox, 1, 1)

        # Descriptive size options
        # Descriptive Size Checkboxes Section
        descriptive_frame = QFrame()
        descriptive_layout = QGridLayout(descriptive_frame)

        self.d32_checkbox = QCheckBox("d32")
        self.dmean_checkbox = QCheckBox("d mean")
        self.dxy_checkbox = QCheckBox("dxy")

        # Add input boxes for `x` and `y` values
        self.dxy_x_input = QLineEdit()
        self.dxy_x_input.setText("5")  # Set default value for x
        self.dxy_x_input.setMaximumWidth(40)

        self.dxy_y_input = QLineEdit()
        self.dxy_y_input.setText("4")  # Set default value for y
        self.dxy_y_input.setMaximumWidth(40)

        # Add elements to the descriptive layout
        descriptive_layout.addWidget(self.d32_checkbox, 0, 0)
        descriptive_layout.addWidget(self.dmean_checkbox, 1, 0)
        descriptive_layout.addWidget(self.dxy_checkbox, 2, 0)
        descriptive_layout.addWidget(QLabel("x"), 2, 1)
        descriptive_layout.addWidget(self.dxy_x_input, 2, 2)
        descriptive_layout.addWidget(QLabel("y"), 2, 3)
        descriptive_layout.addWidget(self.dxy_y_input, 2, 4)

        # Add Save button
        save_frame = QFrame()
        save_layout = QVBoxLayout(save_frame)

        # Folder selection box with button
        folder_selection_frame = QFrame()
        folder_selection_layout = QHBoxLayout(folder_selection_frame)
        self.save_folder_edit = QLineEdit()
        self.save_folder_edit.setPlaceholderText("No folder selected")
        self.save_folder_edit.setReadOnly(True)
        self.save_folder_edit.setMaximumWidth(300)

        select_folder_button = QPushButton("Select Folder")
        select_folder_button.clicked.connect(self.select_save_folder)

        folder_selection_layout.addWidget(self.save_folder_edit)
        folder_selection_layout.addWidget(select_folder_button)

        # Graph filename input row
        graph_frame = QFrame()
        graph_filename_layout = QHBoxLayout(graph_frame)
        current_date = datetime.now().strftime("%Y%m%d")

        self.graph_filename_edit = QLineEdit(
            current_date
        )  # Default name as current date
        graph_filename_label = QLabel(".png")
        graph_filename_layout.addWidget(QLabel("Graph Filename:"))
        graph_filename_layout.addWidget(self.graph_filename_edit)
        graph_filename_layout.addWidget(graph_filename_label)

        # CSV filename input row
        csv_filename_frame = QFrame()
        csv_filename_layout = QHBoxLayout(csv_filename_frame)
        self.csv_filename_edit = QLineEdit(current_date)  # Default name as current date
        csv_filename_label = QLabel(".csv")
        csv_filename_layout.addWidget(QLabel("CSV Filename:"))
        csv_filename_layout.addWidget(self.csv_filename_edit)
        csv_filename_layout.addWidget(csv_filename_label)

        # Save button
        save_button = QPushButton("Save graph and data")
        save_button.clicked.connect(self.save_results)

        # Add folder selection and save button to the layout
        save_layout.addWidget(folder_selection_frame)
        save_layout.addWidget(graph_frame)
        save_layout.addWidget(csv_filename_frame)
        save_layout.addWidget(save_button)

        # Assemble controls layout
        controls_layout.addWidget(histogram_by_label)
        controls_layout.addWidget(self.histogram_by)
        controls_layout.addWidget(self.pdf_checkbox)
        controls_layout.addWidget(self.cdf_checkbox)
        controls_layout.addWidget(bins_label)
        controls_layout.addWidget(self.bins_spinbox)
        controls_layout.addWidget(x_axis_limits_label)
        controls_layout.addWidget(self.min_x_axis_input)
        controls_layout.addWidget(self.max_x_axis_input)
        controls_layout.addWidget(legend_label)
        controls_layout.addWidget(legend_frame)
        controls_layout.addWidget(descriptive_frame)
        controls_layout.addWidget(save_frame)

        # Place graph and controls in the layout
        layout.addWidget(self.histogram_canvas, 0, 0)
        layout.addWidget(controls_frame, 0, 1)

        # Button to generate graph
        generate_button = QPushButton("Generate Graph")
        generate_button.clicked.connect(self.generate_histogram)
        layout.addWidget(generate_button, 1, 0, 1, 2)

        # Add a label to display the descriptive sizes
        self.descriptive_size_label = QLabel("")
        layout.addWidget(self.descriptive_size_label, 2, 0, 1, 2)

    def select_save_folder(self) -> None:
        """Opens a QFileDialog to select a folder for saving."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save")
        if folder_path:
            self.save_folder_edit.setText(folder_path)

    def save_results(self) -> None:
        """Saves histogram and data to the selected folder."""
        folder_path = self.save_folder_edit.text()
        if folder_path == "" or not os.path.exists(folder_path):
            QMessageBox.warning(
                self, "No Folder Selected", "Please select a folder to save the files."
            )
            return

        # Get the user-specified filenames
        graph_filename = self.graph_filename_edit.text()
        csv_filename = self.csv_filename_edit.text()

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
        self.histogram_canvas.fig.savefig(graph_path)

        headers = [
            "area",
            "equivalent_diameter",
            "eccentricity",
            "solidity",
            "circularity",
            "surface_diameter",
        ]
        # Save the CSV data
        rows = []
        for image_properties in self.all_properties:
            for circle in image_properties:
                rows.append(
                    [
                        circle["area"],
                        circle["equivalent_diameter"],
                        circle["eccentricity"],
                        circle["solidity"],
                        circle["circularity"],
                        circle["surface_diameter"],
                    ]
                )

        # Write the data into a CSV file
        with open(csv_path, mode="w", newline="") as data_file:
            writer = csv.writer(data_file)

            # Write the header
            writer.writerow(headers)

            # Write the rows of data
            writer.writerows(rows)

        QMessageBox.information(
            self, "Save Successful", f"Files saved to {folder_path}"
        )
        return

    def generate_histogram(self) -> None:
        """Generate a histogram of equivalent diameters of all detected bubbles.

        This function takes the following steps:
        1. Collect all equivalent diameters from the properties of the detected bubbles
        2. Plot histogram of the equivalent diameters
        3. Calculate descriptive sizes (d32, dmean, dxy)
        4. Update descriptive size label
        5. Optionally add CDF and/or PDF to the histogram
        6. Optionally add vertical lines for the descriptive sizes to the histogram
        7. Add legend to the graph
        8. Redraw the canvas

        :return: None
        """
        # Get settings
        num_bins = self.bins_spinbox.value()
        show_pdf = self.pdf_checkbox.isChecked()
        show_cdf = self.cdf_checkbox.isChecked()
        show_d32 = self.d32_checkbox.isChecked()
        show_dmean = self.dmean_checkbox.isChecked()
        show_dxy = self.dxy_checkbox.isChecked()

        # Collect all equivalent diameters from the properties
        equivalent_diameters_list: list[float] = []

        for circle in self.all_properties:
            equivalent_diameters_list.append(circle["equivalent_diameter"])
        equivalent_diameters_array = np.array(equivalent_diameters_list)

        x_min = float(np.min(equivalent_diameters_array))
        x_max = float(np.max(equivalent_diameters_array))

        # Clear current graph
        self.histogram_canvas.axes.set_xlabel("")
        self.histogram_canvas.axes.set_ylabel("")
        self.histogram_canvas.axes.clear()
        try:
            if self.histogram_canvas.axes2:
                self.histogram_canvas.axes2.clear()
                self.histogram_canvas.axes2.set_ylabel("")
                self.histogram_canvas.axes2.set_yticklabels([])
                self.histogram_canvas.axes2.set_yticks([])
                del self.histogram_canvas.axes2
        except AttributeError:
            pass

        # Plot histogram
        counts, bins, patches = self.histogram_canvas.axes.hist(
            equivalent_diameters_array, bins=num_bins, range=(x_min, x_max)
        )
        # Set graph labels
        self.histogram_canvas.axes.set_xlabel("Equivalent diameter [mm]")
        self.histogram_canvas.axes.set_ylabel("Count [#]")

        # Calculate descriptive sizes
        d32, d_mean, dxy = self.calculate_descriptive_sizes(equivalent_diameters_array)

        # Update descriptive size label
        desc_text = (
            f"Results:\nd32 = {d32:.2f} mm\ndmean = {d_mean:.2f} mm\ndxy = {dxy:.2f} mm"
        )
        self.descriptive_size_label.setText(desc_text)

        # Optionally add CDF
        if show_pdf or show_cdf:
            self.histogram_canvas.axes2 = self.histogram_canvas.axes.twinx()
            self.histogram_canvas.axes2.set_ylabel("Probability [%]")

            if show_cdf:
                cdf = np.cumsum(counts) / np.sum(counts) * 100
                self.histogram_canvas.axes2.plot(
                    bins[:-1], cdf, "r-", marker="o", label="CDF"
                )

            if show_pdf:
                pdf = counts / np.sum(counts) * 100
                self.histogram_canvas.axes2.plot(
                    bins[:-1], pdf, "b-", marker="o", label="PDF"
                )

        if show_d32:
            self.histogram_canvas.axes.axvline(
                x=d32, color="r", linestyle="-", label="d32"
            )

        if show_dmean:
            self.histogram_canvas.axes.axvline(
                x=d_mean, color="g", linestyle="--", label="dmean"
            )

        if show_dxy:
            self.histogram_canvas.axes.axvline(
                x=dxy, color="b", linestyle="--", label="dxy"
            )

        # Apply Legend Options
        legend_position = self.legend_position_combobox.currentText()
        legend_orientation = self.legend_orientation_combobox.currentText()

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
            lines1, labels1 = self.histogram_canvas.axes.get_legend_handles_labels()
            if show_cdf or show_pdf:
                lines2, labels2 = (
                    self.histogram_canvas.axes2.get_legend_handles_labels()
                )
                legend = self.histogram_canvas.axes.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc=legend_location_map.get(legend_position, "upper right"),
                )
            else:
                legend = self.histogram_canvas.axes.legend(
                    lines1,
                    labels1,
                    loc=legend_location_map.get(legend_position, "upper right"),
                )

            if legend_orientation == "Horizontal":
                legend.set_bbox_to_anchor(
                    (1, 1)
                )  # Set orientation of the legend to horizontal if selected

        # Redraw the canvas
        self.histogram_canvas.draw()

        return

    def calculate_descriptive_sizes(
        self, equivalent_diameters: np.ndarray
    ) -> tuple[float, float, float]:
        """Calculate d32, d mean, and dxy based on the equivalent diameters."""
        dxy_x_power: int = int(self.dxy_x_input.text())
        dxy_y_power: int = int(self.dxy_y_input.text())
        d32 = np.sum(equivalent_diameters**3) / np.sum(equivalent_diameters**2)
        # d32, Sauter diameter, should be calculated based on the area, and volume
        # diameter of a circle, which is unkown right now
        d_mean = np.mean(equivalent_diameters)
        dxy = np.sum(equivalent_diameters**dxy_x_power) / np.sum(
            equivalent_diameters**dxy_y_power
        )

        return d32, d_mean, dxy


def main() -> None:
    """Start the GUI application.

    This function initializes the PySide6 application and displays the main window.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# GUI - Jump generated graph after processing & every choice of additional elements
# Detection and filtering - Seperate process from filtering
# Store last previewed images
