"""Component handlers module for the Bubble Analyser application.

This module provides model classes and handlers for managing various components of the
application, including image processing, calibration, and data management. It serves as
the bridge between the GUI interface and the underlying processing functionality.

The module contains the following key components:
- WorkerThread: A thread class for handling background processing tasks
- InputFilesModel: Model for managing input image files and paths
- CalibrationModel: Model for managing calibration data and pixel-to-millimeter
    conversion
- ImageProcessingModel: Model for managing image processing operations and parameters

These components work together to provide a structured approach to image processing,
data management, and user interface interaction in the Bubble Analyser application.
"""

import logging
import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy import typing as npt
from PySide6.QtCore import QEventLoop, QThread, Signal

from bubble_analyser.processing import (
    Config,
    EllipseAdjuster,
    FilterParamHandler,
    Image,
    MethodsHandler,
    calculate_px2mm,
)


class WorkerThread(QThread):
    """A worker thread class for handling batch image processing operations.

    This class extends QThread to perform image processing tasks in the background,
    preventing the GUI from freezing during lengthy operations. It provides progress
    updates and completion signals.

    Attributes:
        update_progress (Signal[int]): Signal emitted to update the progress bar.
        processing_done (Signal): Signal emitted when processing is complete.
        if_save (bool): Flag indicating whether to save processed images.
        save_path (Path): Directory path where processed images should be saved.
        model (ImageProcessingModel): The model containing image processing logic.
    """

    update_progress = Signal(int)
    processing_done = Signal()

    def __init__(  # type: ignore
        self,
        model,
        if_save_processed_image: bool = False,
        save_path: Path = cast(Path, None),
    ) -> None:
        """Initialize the worker thread with processing parameters.

        Args:
            model (ImageProcessingModel): The model containing image processing logic.
            if_save_processed_image (bool, optional): Whether to save processed images.
                Defaults to False.
            save_path (Path, optional): Directory to save processed images.
                Defaults to None.
        """
        super().__init__()
        self.if_save = if_save_processed_image
        self.save_path = save_path
        self.model: ImageProcessingModel = model

    def run(self) -> None:
        """Execute the batch processing operation.

        This method is called when the thread starts. It delegates the actual processing
        to the model's batch_process_images method.
        """
        self.model.batch_process_images(self, self.if_save, self.save_path)

    def update_progress_bar(self, value: int) -> None:
        """Emit a signal to update the progress bar in the GUI.

        Args:
            value (int): The current progress value to display.
        """
        self.update_progress.emit(value)

    def on_processing_done(self) -> None:
        """Emit a signal indicating that processing is complete.

        This method is called when all images have been processed.
        """
        self.processing_done.emit()


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
        # if folder_path is None:
        #     folder_path = self.folder_path

        self.image_list = []
        self.image_list_full_path = []

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self.image_list.append(file_name)
                self.image_list_full_path.append(os.path.join(folder_path, file_name))
        return self.image_list, self.image_list_full_path


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

        self.px2mm: float
        self.px2mm_display: float
        self.calibration_confirmed: bool = False

    def get_px2mm_ratio(  # type: ignore
        self,
        pixel_img_path: Path,
        img_resample: float = 0.5,
        gui=None,  # type: ignore
    ) -> tuple[float, MatLike]:
        """Calculate the pixel-to-millimeter ratio from a calibration image.

        This method uses the calculate_px2mm function to determine the conversion
        ratio between pixels and millimeters based on a calibration image
        (typically containing a ruler).

        Args:
            pixel_img_path (Path): Path to the calibration image file.
            img_resample (float, optional): Resampling factor for the image.
                Defaults to 0.5.
            gui (object, optional): GUI object for displaying interactive elements.
                Defaults to None.

        Returns:
            float: The calculated pixel-to-millimeter ratio.
            img_drawed_line: The ruler image with the drawn line.
        """
        __, self.px2mm, img_drawed_line = calculate_px2mm(pixel_img_path, img_resample, gui)  # type: ignore

        return self.px2mm, img_drawed_line

    def confirm_calibration(self) -> None:
        """Mark the calibration as confirmed.

        This method sets the calibration_confirmed flag to True, indicating that
        the calibration process has been completed and confirmed by the user.
        """
        self.calibration_confirmed = True


class ImageProcessingModel:
    """A model class for managing image processing operations and parameters.

    This class handles the processing of images using various algorithms, maintains
    processing parameters, and manages the state of processed images. It interfaces
    with the MethodsHandler to access processing routines and stores the results of
    image processing operations.

    Attributes:
        algorithm (str): The currently selected processing algorithm name.
        params (Config): Configuration parameters for image processing.
        filter_param_dict (dict[str, float]): Dictionary of filtering parameters.
        px2mm (float): Conversion factor from pixels to millimeters.
        if_bknd (bool): Flag indicating whether a background image is being used.
        bknd_img_path (Path): Path to the background image file.
        img_path_list (list[Path]): List of paths to images to be processed.
        img_dict (dict[Path, Image]): Dictionary mapping image paths to their Image
            objects.
        adjuster (EllipseAdjuster): Tool for manual adjustment of detected ellipses.
        ellipses_properties (list[list[dict[str, float]]]): Properties of detected
            ellipses
        for all images. methods_handler (MethodsHandler): Handler for accessing
            processing methods.
        all_methods_n_params (dict): Dictionary of all available methods and their
            parameters.
    """

    def __init__(self, params: Config) -> None:
        """Initialize the ImageProcessingModel with the provided configuration.

        Args:
            params (Config): Configuration parameters for image processing.
        """
        super().__init__()

        self.algorithm: str = ""
        self.params_config: Config = params

        self.filter_param_dict_1: dict[str, float | str]
        self.filter_param_dict_2: dict[str, float | str]

        self.px2mm_display: float
        self.if_bknd: bool
        self.bknd_img_path: Path = cast(Path, None)

        self.img_path_list: list[Path] = []
        self.img_dict: dict[Path, Image] = {}

        self.adjuster: EllipseAdjuster
        self.ellipses_properties: list[list[dict[str, float | str]]] = []

        logging.info("------------------------------Intializing Parameters------------------------------")
        self.methods_handler: MethodsHandler
        self.filter_param_handler: FilterParamHandler
        self.initialize_methods_handlers()
        self.initialize_filter_param_handler()

    def initialize_methods_handlers(self) -> None:
        """Initialize the methods handler and retrieve available processing methods.

        This method creates a new MethodsHandler instance using the current
        configuration and retrieves the dictionary of available processing methods
        and their parameters.
        """
        self.methods_handler = MethodsHandler(self.params_config)
        self.all_methods_n_params = self.methods_handler.full_dict
        logging.info(f"All detected methods and their parameters: {self.all_methods_n_params}")

    def initialize_filter_param_handler(self) -> None:
        """Initialize the filter parameter handler and retrieve filtering parameters.

        This method creates a new FilterParamHandler instance using the current
        configuration parameters and retrieves the dictionary of needed filtering
        parameters. The filter parameters are used to control various aspects of
        the image processing pipeline such as thresholds, sizes, and other
        filtering criteria.

        The parameters are also printed to the console for debugging purposes.
        """
        self.filter_param_handler = FilterParamHandler(self.params_config.model_dump())
        self.filter_param_dict_1, self.filter_param_dict_2 = self.filter_param_handler.get_needed_params()
        logging.info(f"Basic filtering parameters: {self.filter_param_dict_1}")
        logging.info(f"Find circles filtering parameters: {self.filter_param_dict_2}")

    def confirm_folder_selection(self, folder_path_list: list[Path]) -> None:
        """Set the list of image paths to be processed.

        Args:
            folder_path_list (list[Path]): List of paths to images to be processed.
        """
        self.img_path_list = folder_path_list

    def get_bknd_img_path(self, bknd_img_path: Path) -> None:
        """Set the path to the background image.

        Args:
            bknd_img_path (Path): Path to the background image file.
        """
        self.bknd_img_path = Path(bknd_img_path)

    def update_px2mm_display(self, px2mm_display: float) -> None:
        """Update the pixel-to-millimeter conversion ratio.

        Args:
            px2mm_display (float): The new pixel-to-millimeter conversion ratio for display.
        """
        self.px2mm_display = px2mm_display

    def preview_processed_image(self, index: int) -> tuple[bool, npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Retrieve the processed images for preview.

        Args:
            index (int): Index of the image in the image list.

        Returns:
            tuple[bool, npt.NDArray[np.int_], npt.NDArray[np.int_]]: A tuple containing:
                - Boolean indicating if the image exists and has been processed
                - The image with labels before filtering
                - The image with ellipses overlaid after filtering
        """
        name = self.img_path_list[index]
        if_img = False
        img_before_filter = cast(npt.NDArray[np.int_], None)
        img_after_filter = cast(npt.NDArray[np.int_], None)
        if name in self.img_dict:
            img_before_filter = self.img_dict[name].labels_on_img_before_filter

            try:
                img_after_filter = self.img_dict[name].ellipses_on_images
            except AttributeError as e:
                print(e)
                img_after_filter = cast(npt.NDArray[np.int_], None)
                return (
                    False,
                    cast(npt.NDArray[np.int_], None),
                    cast(npt.NDArray[np.int_], None),
                )
            if_img = True

        return if_img, img_before_filter, img_after_filter

    def load_filter_params(self, dict_params_1: dict[str, float | str], dict_params_2: dict[str, float | str]) -> None:
        """Load filtering parameters into the model.

        Args:
            dict_params_1 (dict[str, float]): Dictionary containing filtering parameters.
            dict_params_2 (dict[str, float]): Dictionary containing find circles parameters.
        """
        self.filter_param_dict_1 = dict_params_1
        self.filter_param_dict_2 = dict_params_2

    def initialize_image(self, name: Path) -> None:
        """Initialize an Image object for processing if it doesn't already exist.

        Args:
            name (Path): Path to the image file.
        """
        if name not in self.img_dict:
            self.img_dict[name] = Image(
                self.px2mm_display,
                raw_img_path=cast(Path, name),
                all_methods_n_params=self.all_methods_n_params,
                methods_handler=self.methods_handler,
                bknd_img_path=self.bknd_img_path,
            )

    def step_1_main(self, index: int) -> npt.NDArray[np.int_]:
        """Execute the first step of image processing (pre-filtering).

        Args:
            index (int): Index of the image to process.

        Returns:
            npt.NDArray[np.int_]: The processed image with labels before filtering.
        """
        name = self.img_path_list[index]
        self.initialize_image(name)

        self.img_dict[name].processing_image_before_filtering(self.algorithm)
        return self.img_dict[name].labels_on_img_before_filter

    def step_2_main(self, index: int) -> npt.NDArray[np.int_]:
        """Execute the second step of image processing (filtering&ellipse detection).

        Args:
            index (int): Index of the image to process.

        Returns:
            npt.NDArray[np.int_]: The processed image with detected ellipses overlaid.
        """
        name = self.img_path_list[index]
        self.img_dict[name].load_filter_params(self.filter_param_dict_1, self.filter_param_dict_2)
        self.img_dict[name].filtering_processing()

        return self.img_dict[name].ellipses_on_images

    def ellipse_manual_adjustment(self, index: int) -> npt.NDArray[np.int_]:
        """Launch the ellipse adjustment tool for manual fine-tuning of ellipses.

        This method creates an EllipseAdjuster instance for the specified image,
        displays the adjustment interface, and waits for the user to complete the
        adjustments.

        Args:
            index (int): Index of the image to adjust.

        Returns:
            npt.NDArray[np.int_]: The updated image with adjusted ellipses overlaid.
        """
        logging.info("Ellipse handler triggered.")
        name = self.img_path_list[index]
        image = self.img_dict[name]
        self.adjuster = EllipseAdjuster(image.ellipses, image.img_rgb)

        loop = QEventLoop()

        def on_finished() -> None:
            self.handle_ellipse_adjustment_finished(image)
            loop.quit()

        self.adjuster.finished.connect(on_finished)
        self.adjuster.show()

        loop.exec()
        logging.info("Ellipse handler finished.")
        return image.ellipses_on_images

    def handle_ellipse_adjustment_finished(self, image: Image) -> None:
        """Process the results of manual ellipse adjustment.

        This method is called when the user completes the manual adjustment process.
        It updates the image's ellipses with the adjusted ones and regenerates
        the overlay.

        Args:
            image (Image): The image object containing the ellipses to update.
        """
        image.update_ellipses(self.adjuster.ellipses)
        image.overlay_ellipses_on_images()

    def batch_process_images(
        self,
        worker_thread: WorkerThread,
        if_save: bool,
        save_path: Path = cast(Path, None),
    ) -> None:
        """Process all images in the image list using the current parameters.

        This method iterates through all images in the image list, applies the current
        processing parameters, and optionally saves the processed images. It updates
        the progress through the provided worker thread.

        Args:
            worker_thread (WorkerThread): Thread object for progress reporting.
            if_save (bool): Whether to save the processed images.
            save_path (Path, optional): Directory to save processed images.
                Defaults to None.
        """
        # Process every image in the list
        for index, name in enumerate(self.img_path_list):
            logging.info("------------------------------Batch Process Started------------------------------")
            logging.info(f"If saving processed images: {if_save}")
            self.initialize_image(name)
            self.img_dict[name].load_filter_params(self.filter_param_dict_1, self.filter_param_dict_2)

            # Check if the image has been manually fine tuned
            if self.img_dict[name].if_fine_tuned:
                # If fine tuned, save the ellipses properties
                # and skip the processing
                logging.info(f"This image has been fine tuned: {name}, no need to process again.")
                self.img_dict[name].get_ellipse_properties()
                self.ellipses_properties.append(self.img_dict[name].ellipses_properties)

                if if_save:
                    self.save_processed_images(self.img_dict[name].ellipses_on_images, name, save_path)
                    self.save_labelled_masks(self.img_dict[name].labelled_ellipses_mask, name, save_path)
                    continue

            self.img_dict[name].processing_image_before_filtering(self.algorithm)
            self.img_dict[name].filtering_processing()

            self.ellipses_properties.append(self.img_dict[name].ellipses_properties)

            if if_save:
                self.save_processed_images(self.img_dict[name].ellipses_on_images, name, save_path)
                self.save_labelled_masks(self.img_dict[name].labelled_ellipses_mask, name, save_path)

            worker_thread.update_progress_bar(index + 1)
        worker_thread.on_processing_done()

    def save_processed_images(self, img: npt.NDArray[np.int_], img_name: Path, save_path: Path) -> None:
        """Save the processed image with detected ellipses to disk.

        Args:
            img (npt.NDArray[np.int_]): The processed image array to save.
            img_name (Path): Original image path used to generate the output filename.
            save_path (Path): Directory where the image should be saved.
        """
        file_name = os.path.basename(img_name)
        new_name = os.path.join(save_path, file_name)
        logging.info(f"Processed image with ellipses saving to: {new_name}")
        try:
            cv2.imwrite(new_name, img)
            logging.info("saved")
        except Exception as e:
            logging.info(e)

    def save_labelled_masks(self, img: npt.NDArray[np.int_], img_name: Path, save_path: Path) -> None:
        """Save the labelled mask image to disk.

        This method saves a mask image where each detected ellipse is labeled with a
        unique identifier.
        The output filename is the original filename with '_mask.png' appended.

        Args:
            img (npt.NDArray[np.int_]): The mask image array to save.
            img_name (Path): Original image path used to generate the output filename.
            save_path (Path): Directory where the mask should be saved.
        """
        file_name = os.path.basename(img_name)
        new_name = os.path.join(save_path, f"{file_name}_mask.png")
        logging.info(f"Labelled mask saving to: {new_name}")
        try:
            cv2.imwrite(new_name, img)
            logging.info("saved")
        except Exception as e:
            logging.info(e)
