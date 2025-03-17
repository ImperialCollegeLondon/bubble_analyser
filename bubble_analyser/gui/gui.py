"""GUI module for the Bubble Analyser application.

This module implements the graphical user interface for the Bubble Analyser application,
providing a user-friendly interface for analyzing bubble images. It defines the main
window and all UI components including tabs for folder selection, calibration, image
processing, and results visualization.

The module contains the following key components:
- MplCanvas: A class for embedding Matplotlib figures in the PySide6 application
- MainWindow: The main application window with multiple tabs for different functionalities

The GUI is structured around a tab-based interface with separate sections for:
1. Folder selection - for choosing input image folders
2. Calibration - for pixel-to-mm calibration and background image selection
3. Image processing - for bubble detection, filtering, and manual adjustment
4. Results - for visualizing and exporting histogram data

The module works closely with event handlers defined in the event_handlers module
to connect UI interactions with the underlying processing functionality.
"""

from datetime import datetime

import matplotlib

import bubble_analyser.gui.event_handlers as hd

matplotlib.use("Agg")  # Force backend to Agg for CI
from typing import cast

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class MplCanvas(FigureCanvas):
    """A class for creating a Matplotlib figure within a PySide6 application.

    Attributes:
        fig: The Matplotlib figure.
        axes: The axes of the figure.
    """

    def __init__(self, parent: QMainWindow, width: float = 5, height: float = 4, dpi: float = 100) -> None:
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
    """The main window of the Bubble Analyser application.

    This class represents the main window of the application, containing multiple tabs
    for different functionalities including folder selection, calibration, image
    processing, and results visualization. It manages the overall layout and
    interaction between different components of the application.

    Attributes:
        main_handler: The main event handler for the application.
        folder_tab_handler: Handler for folder selection tab operations.
        calibration_tab_handler: Handler for calibration tab operations.
        image_processing_tab_handler: Handler for image processing tab operations.
        results_tab_handler: Handler for results tab operations.
        bknd_img_exist: Flag indicating if background image exists.
        calibration_confirmed: Flag indicating if calibration is confirmed.
    """

    def __init__(self, main_handler: hd.MainHandler) -> None:
        """Initialize the main window of the Bubble Analyser application.

        Sets up the main window with all its components including the menu bar
        and different tabs for folder selection, calibration, image processing,
        and results visualization.

        Args:
            main_handler: The main event handler that manages the application's logic.
        """
        super().__init__()

        self.main_handler: hd.MainHandler = main_handler

        self.folder_tab_handler = self.main_handler.folder_tab_handler

        self.calibration_tab_handler = self.main_handler.calibration_tab_handler

        self.image_processing_tab_handler = self.main_handler.image_processing_tab_handler

        self.results_tab_handler = self.main_handler.results_tab_handler

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

        # Add Calibration Tab
        self.calibration_tab = QWidget()
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.bg_image_confirmed = False
        self.px_res_confirmed = False

        # Add Image Processing Tab
        self.current_algorithm: str = cast(str, None)
        self.image_processing_tab = QWidget()
        self.tabs.addTab(self.image_processing_tab, "Bubble detection and filtering")

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.setup_menu_bar()

        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Results")

    def load_full_gui(self) -> None:
        """Load and initialize all GUI components.

        Sets up all the tabs in the application including the folder selection,
        calibration, image processing, and results tabs. This method should be
        called after the main window is created to complete the GUI initialization.
        """
        self.setup_folder_tab()
        self.setup_calibration_tab()
        self.setup_image_processing_tab()
        self.setup_results_tab()

    def setup_menu_bar(self) -> None:
        """Set up the application's menu bar.

        Creates and configures the menu bar with settings menu items including
        options to export settings and restart a new session.
        """
        setting_menu = QMenu("Settings")
        export_setting_action = QAction("Export settings", self)
        export_setting_action.triggered.connect(self.main_handler.menubar_open_export_settings_dialog)
        setting_menu.addAction(export_setting_action)

        restart_action = QAction("Restart a new session", self)
        restart_action.triggered.connect(self.main_handler.menubar_ask_if_restart)
        setting_menu.addAction(restart_action)

        setting_menu = self.menu_bar.addMenu(setting_menu)  # type: ignore

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
        self.folder_path_edit.setText(str(self.folder_tab_handler.image_path))
        select_folder_button = QPushButton("Select Folder")
        confirm_folder_button = QPushButton("Confirm Folder")
        select_folder_button.clicked.connect(self.main_handler.tab1_select_folder)
        confirm_folder_button.clicked.connect(self.main_handler.tab1_confirm_folder_selection)
        top_layout.addWidget(select_folder_button)
        top_layout.addWidget(self.folder_path_edit)
        top_layout.addWidget(confirm_folder_button)

        # Bottom Left: List of images in folder
        bottom_left_frame = QFrame()
        bottom_left_layout = QVBoxLayout(bottom_left_frame)
        self.image_list = QListWidget()
        self.image_list.clicked.connect(self.folder_tab_handler.preview_image_folder_tab)
        bottom_left_layout.addWidget(self.image_list)

        # Bottom Right: Image Preview
        bottom_right_frame = QFrame()
        bottom_right_layout = QVBoxLayout(bottom_right_frame)
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFixedSize(600, 600)  # Set a fixed size for the image preview
        bottom_right_layout.addWidget(self.image_preview)

        # Split the bottom part into two sections
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.addWidget(bottom_left_frame)
        bottom_layout.addWidget(bottom_right_frame)

        # Add top and bottom frames to the main layout
        layout.addWidget(top_frame, 1)
        layout.addWidget(bottom_frame, 6)

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
        pixel_img_frame = QFrame()
        pixel_img_layout = QVBoxLayout(pixel_img_frame)
        self.pixel_img_name = QLineEdit()
        self.pixel_img_name.setText(str(self.calibration_tab_handler.px_img_path))
        pixel_img_confirm_button = QPushButton("Confirm")
        pixel_img_confirm_button.clicked.connect(self.main_handler.tab2_get_px2mm_ratio)
        pixel_img_select_button = QPushButton("Select other image for resolution calibration")
        pixel_img_select_button.clicked.connect(self.main_handler.tab2_select_px_mm_image)
        self.pixel_img_preview = QLabel()
        self.pixel_img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixel_img_preview.setFixedSize(400, 300)

        pixel_img_layout.addWidget(QLabel("Step 1: Pixel resolution calibration"))
        pixel_img_layout.addWidget(QLabel("Image name"))
        pixel_img_layout.addWidget(self.pixel_img_name)
        pixel_img_layout.addWidget(pixel_img_confirm_button)
        pixel_img_layout.addWidget(pixel_img_select_button)
        pixel_img_layout.addWidget(self.pixel_img_preview)
        pixel_img_layout.addStretch()

        # Background Correction Image
        bg_corr_frame = QFrame()
        bg_corr_layout = QVBoxLayout(bg_corr_frame)
        self.bg_corr_image_name = QLineEdit()
        self.bg_corr_image_name.setText("Choose your background image from local")
        # bg_corr_confirm_button = QPushButton("Confirm")
        # bg_corr_confirm_button.clicked.connect(self.confirm_bg_corr_image)
        bg_corr_select_button = QPushButton("Select other image for background correction")
        bg_corr_select_button.clicked.connect(self.main_handler.tab2_select_bg_corr_image)
        self.bg_corr_image_preview = QLabel()
        self.bg_corr_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bg_corr_image_preview.setFixedSize(400, 300)

        bg_corr_layout.addWidget(QLabel("Step 2: Background correction image"))
        bg_corr_layout.addWidget(QLabel("Image name"))
        bg_corr_layout.addWidget(self.bg_corr_image_name)
        # bg_corr_layout.addWidget(bg_corr_confirm_button)
        bg_corr_layout.addWidget(bg_corr_select_button)
        bg_corr_layout.addWidget(self.bg_corr_image_preview)
        bg_corr_layout.addStretch()

        # Adding Pixel Resolution and Background Correction frames to top layout
        top_frame_layout.addWidget(pixel_img_frame)
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
            self.main_handler.tab2_confirm_calibration
        )  # Connect confirm button to the handler

        bottom_frame_layout.addWidget(manualcalibration_frame)
        bottom_frame_layout.addWidget(confirm_px_mm_button)

        # Add frames to main layout
        layout.addWidget(top_frame, 0, 0, 1, 2)
        layout.addWidget(bottom_frame, 1, 0, 1, 2)

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

        # ----------- First Column: Sample Image Preview -----------

        first_column_frame = QFrame()
        first_column_layout = QVBoxLayout(first_column_frame)

        # Sample Image Preview Canvas
        self.sample_image_preview = QLabel("Sample image preview")
        self.sample_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_image_preview.setFixedSize(400, 300)  # Adjust size as needed

        prev_button = QPushButton("< Prev. Img")
        next_button = QPushButton("Next Img >")
        prev_button.clicked.connect(lambda: self.main_handler.tab3_update_sample_image("prev"))
        next_button.clicked.connect(lambda: self.main_handler.tab3_update_sample_image("next"))
        preview_processed_images_button = QPushButton("Preview Processed Images")
        preview_processed_images_button.clicked.connect(self.main_handler.tab3_preview_processed_images)

        first_column_layout.addWidget(QLabel("Select image and preview"))
        first_column_layout.addWidget(self.sample_image_preview)

        # Prev/Next Buttons
        first_column_buttons_layout = QHBoxLayout()
        first_column_buttons_layout.addWidget(prev_button)
        first_column_buttons_layout.addWidget(next_button)

        first_column_layout.addLayout(first_column_buttons_layout)
        first_column_layout.addWidget(preview_processed_images_button)

        # ----------- Second Column: Processed Image Before Filtering and Sandbox ----

        second_column_frame = QFrame()
        second_column_layout = QVBoxLayout(second_column_frame)

        # Processed Image Before Filtering Canvas
        self.label_before_filtering = MplCanvas(self, width=5, height=4, dpi=100)
        second_column_layout.addWidget(QLabel("Step 1:Processed Image_Before Filtering"))
        second_column_layout.addWidget(self.label_before_filtering)

        algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_combo = QComboBox()
        self.image_processing_tab_handler.initialize_algorithm_combo()
        self.algorithm_combo.currentTextChanged.connect(
            lambda: self.main_handler.tab3_handle_algorithm_change(self.algorithm_combo.currentText())
        )

        # Parameter sandbox for img_resample_factor, threshold_value, element_size
        sandbox1_label = QLabel("Adjust parameters before filtering")

        second_column_layout.addWidget(algorithm_label)
        second_column_layout.addWidget(self.algorithm_combo)
        second_column_layout.addWidget(sandbox1_label)

        self.param_sandbox1 = QTableWidget(0, 2)
        self.param_sandbox1.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.main_handler.tab3_load_parameter_table_1(self.algorithm_combo.currentText())  # New helper method

        # Confirm button for this sandbox
        preview_button1 = QPushButton("Confirm parameter and preview")
        preview_button1.clicked.connect(self.main_handler.tab3_confirm_parameter_before_filtering)

        second_column_layout.addWidget(sandbox1_label)
        second_column_layout.addWidget(self.param_sandbox1)
        second_column_layout.addWidget(preview_button1)

        # ----------- Third Column: Processed Image After Filtering and Sandbox ------

        third_column_frame = QFrame()
        third_column_layout = QVBoxLayout(third_column_frame)

        # Processed Image After Filtering Canvas
        self.processed_image_preview = MplCanvas(self, width=5, height=4, dpi=100)
        third_column_layout.addWidget(QLabel("Step 2: Filtering and Selecting Circles"))
        third_column_layout.addWidget(self.processed_image_preview)

        # Parameter sandbox for max_eccentricity, min_solidity, min_size
        sandbox2_label = QLabel("Adjust parameters for filtering circles")
        self.param_sandbox2 = QTableWidget(0, 2)
        self.param_sandbox2.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.main_handler.tab3_initialize_parameter_table_2()  # New helper method

        # Confirm and Batch Process buttons for this sandbox
        preview_button2 = QPushButton("Confirm parameter and preview")
        manual_adjustment_button = QPushButton("Manual adjustment")
        manual_adjustment_button.clicked.connect(self.main_handler.tab3_ellipse_manual_adjustment)
        batch_process_button = QPushButton("Batch process images")
        preview_button2.clicked.connect(self.main_handler.tab3_confirm_parameter_for_filtering)
        batch_process_button.clicked.connect(self.main_handler.tab3_ask_if_batch)

        third_column_layout.addWidget(sandbox2_label)
        third_column_layout.addWidget(self.param_sandbox2)
        third_column_layout.addWidget(preview_button2)
        third_column_layout.addWidget(manual_adjustment_button)
        third_column_layout.addWidget(batch_process_button)

        # Add the columns to the main layout
        layout.addWidget(first_column_frame, 0, 0)
        layout.addWidget(second_column_frame, 0, 1)
        layout.addWidget(third_column_frame, 0, 2)

    def setup_results_tab(self) -> None:
        """Set up the results tab.

        This function sets up the results tab by creating the following widgets:
        1. A graph canvas for displaying the histogram.
        2. Controls for histogram options, including the type of histogram to
        generate (by number or volume), checkboxes for PDF and CDF, the number
        of bins, and the x-axis limits.
        3. A legend position.
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

        # # Histogram type
        # histogram_by_label = QLabel("Histogram by:")
        # self.histogram_by = QComboBox()
        # # self.histogram_by.addItems(["Number", "Volume"])
        # self.histogram_by.addItems(["Number"])
        # self.histogram_by.currentIndexChanged.connect(
        #     self.results_tab_handler.generate_histogram
        # )  # Connect to auto-update

        self.options_label = QLabel("Histogram Options:")
        # PDF/CDF Checkboxes
        self.pdf_checkbox = QCheckBox("PDF")
        self.cdf_checkbox = QCheckBox("CDF")
        self.pdf_checkbox.stateChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update
        self.cdf_checkbox.stateChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update

        # Number of bins
        bins_label = QLabel("Number of bins:")
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setValue(15)
        self.bins_spinbox.setRange(1, 100)
        self.bins_spinbox.valueChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update

        # X-axis limits
        x_axis_limits_label = QLabel("X-axis limits:")
        self.min_x_axis_input = QLineEdit("0.0")
        self.max_x_axis_input = QLineEdit("5.0")
        self.min_x_axis_input.textChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update
        self.max_x_axis_input.textChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update

        legend_label = QLabel("Legend settings:")
        legend_frame = QFrame()
        legend_layout = QGridLayout(legend_frame)

        # Legend position dropdown
        legend_layout.addWidget(QLabel("Position:"), 0, 0)
        self.legend_position_combobox = QComboBox()
        self.legend_position_combobox.addItems(["North East", "North West", "South East", "South West"])
        self.legend_position_combobox.currentIndexChanged.connect(
            self.results_tab_handler.generate_histogram
        )  # Connect to auto-update
        legend_layout.addWidget(self.legend_position_combobox, 0, 1)

        # Descriptive size options
        # Descriptive Size Checkboxes Section
        descriptive_frame = QFrame()
        descriptive_layout = QGridLayout(descriptive_frame)

        self.d32_checkbox = QCheckBox("d32")
        self.dmean_checkbox = QCheckBox("d mean")
        self.dxy_checkbox = QCheckBox("dxy")
        self.d32_checkbox.stateChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update
        self.dmean_checkbox.stateChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update
        self.dxy_checkbox.stateChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update

        # Add input boxes for `x` and `y` values
        self.dxy_x_input = QLineEdit()
        self.dxy_x_input.setText("5")  # Set default value for x
        self.dxy_x_input.setMaximumWidth(40)
        self.dxy_x_input.textChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update

        self.dxy_y_input = QLineEdit()
        self.dxy_y_input.setText("4")  # Set default value for y
        self.dxy_y_input.setMaximumWidth(40)
        self.dxy_y_input.textChanged.connect(self.results_tab_handler.generate_histogram)  # Connect to auto-update

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
        # folder_selection_frame = QFrame()
        # folder_selection_layout = QHBoxLayout(folder_selection_frame)
        # self.save_folder_edit = QLineEdit()
        # self.save_folder_edit.setText(str(self.results_tab_handler.save_path))
        # # self.save_folder_edit.setPlaceholderText("No folder selected")
        # self.save_folder_edit.setReadOnly(True)
        # self.save_folder_edit.setMaximumWidth(300)

        # select_folder_button = QPushButton("Select Folder")
        # select_folder_button.clicked.connect(self.results_tab_handler.select_save_folder)

        # folder_selection_layout.addWidget(self.save_folder_edit)
        # folder_selection_layout.addWidget(select_folder_button)

        # Graph filename input row
        graph_frame = QFrame()
        graph_filename_layout = QHBoxLayout(graph_frame)
        current_date = datetime.now().strftime("%Y%m%d")

        self.graph_filename_edit = QLineEdit(current_date)  # Default name as current date
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
        save_button.clicked.connect(self.results_tab_handler.save_results)

        # Add folder selection and save button to the layout
        # save_layout.addWidget(folder_selection_frame)
        save_layout.addWidget(graph_frame)
        save_layout.addWidget(csv_filename_frame)
        save_layout.addWidget(save_button)

        # Assemble controls layout
        # controls_layout.addWidget(histogram_by_label)
        # controls_layout.addWidget(self.histogram_by)
        controls_layout.addWidget(self.options_label)
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

        # Add a label to display the descriptive sizes
        self.descriptive_size_label = QLabel("")
        layout.addWidget(self.descriptive_size_label, 2, 0, 1, 2)
