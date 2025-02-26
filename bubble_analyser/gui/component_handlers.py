import os
from pathlib import Path

import cv2
import numpy as np
from numpy import typing as npt
from PySide6.QtCore import QEventLoop, QThread, Signal

from bubble_analyser.processing import (
    Config,
    EllipseAdjuster,
    Image,
    MethodsHandler,
    calculate_px2mm,
)


class WorkerThread(QThread):
    update_progress = Signal(int)
    processing_done = Signal()

    def __init__(
        self,
        model,
        if_save_processed_image: bool = False,
        save_path: Path = None,  # type: ignore
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.if_save = if_save_processed_image
        self.save_path = save_path
        self.model: ImageProcessingModel = model

    def run(self) -> None:
        self.model.batch_process_images(self, self.if_save, self.save_path)

    def update_progress_bar(self, value: int) -> None:
        self.update_progress.emit(value)

    def on_processing_done(self) -> None:
        self.processing_done.emit()


class InputFilesModel:
    def __init__(self) -> None:
        self.sample_images_confirmed: bool = False
        self.folder_path: Path = None

        self.image_list: list[str] = []

        # full path for ui event handlers
        self.image_list_full_path: list[str] = []

        # full path for processing models
        self.image_list_full_path_in_path: list[Path] = []
        self.current_image_idx: int = 0

    def confirm_folder_selection(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)
        _ = self.get_image_list(folder_path)

        for path in self.image_list_full_path:
            self.image_list_full_path_in_path.append(Path(path))

        self.sample_images_confirmed = True

    def get_image_list(self, folder_path: str) -> list[str]:
        if folder_path is None:
            folder_path = self.folder_path

        self.image_list = []
        self.image_list_full_path = []

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self.image_list.append(file_name)
                self.image_list_full_path.append(os.path.join(folder_path, file_name))
        return self.image_list, self.image_list_full_path


class CalibrationModel:
    def __init__(self) -> None:
        self.pixel_img_confirmed: bool = False
        self.bknd_img_confirmed: bool = False

        self.bknd_img_path: Path = None
        self.bknd_img: npt.NDArray[np.int_]
        self.if_bknd: bool = False

        self.pixel_img_path: Path
        self.pixel_img: npt.NDArray[np.int_]

        self.px2mm: float
        self.calibration_confirmed: bool = False

    def get_px2mm_ratio(
        self, pixel_img_path: str = None, img_resample: float = 0.5, gui=None
    ) -> float:  # type: ignore
        if pixel_img_path is None:
            pixel_img_path = self.pixel_img_path
        __, self.px2mm = calculate_px2mm(pixel_img_path, img_resample, gui)

        return self.px2mm

    def confirm_calibration(self) -> None:
        self.calibration_confirmed = True


class ImageProcessingModel:
    def __init__(self, params: Config) -> None:
        super().__init__()

        self.algorithm: str = ""
        self.params: Config = params

        self.filter_param_dict = {
            "max_eccentricity": 0.0,
            "min_solidity": 0.0,
            "min_size": 0.0,
        }

        self.px2mm: float
        self.if_bknd: bool
        self.bknd_img_path: Path = None

        self.img_path_list: list[str] = []
        self.img_dict: dict[str, Image] = {}

        self.adjuster: EllipseAdjuster
        self.ellipses_properties: list[list[dict[str, float]]] = []

        self.methods_handler: MethodsHandler
        self.initialize_methods_handlers()

    def initialize_methods_handlers(self) -> None:
        self.methods_handler = MethodsHandler(self.params)
        self.all_methods_n_params = self.methods_handler.full_dict
        print("all_methods_n_params", self.all_methods_n_params)

    def confirm_folder_selection(self, folder_path_list: list[str]) -> None:
        self.img_path_list = folder_path_list

    def get_bknd_img_path(self, bknd_img_path: str) -> Path:
        self.bknd_img_path = Path(bknd_img_path)

    def update_px2mm(self, px2mm: float) -> None:
        self.px2mm = px2mm

    def preview_processed_image(self, index) -> None:
        name = self.img_path_list[index]
        if_img = False
        img_before_filter = None
        img_after_filter = None
        if name in self.img_dict:
            img_before_filter = self.img_dict[name].labels_on_img_before_filter

            try:
                img_after_filter = self.img_dict[name].ellipses_on_images
            except AttributeError as e:
                print(e)
                img_after_filter = None
                return False, None, None
            if_img = True

        return if_img, img_before_filter, img_after_filter

    def load_filter_params(self, dict_params: dict):
        self.filter_param_dict = dict_params

    def processing_image_before_filtering(self) -> None:
        if self.algorithm == "normal_watershed":
            self.img_resample_factor = self.img_resample_factor
            self.threshold_value = self.threshold_value
            self.element_size = self.element_size
            self.connectivity = self.connectivity
        else:
            self.img_resample_factor = self.img_resample_factor
            self.element_size = self.element_size

    def initialize_image(self, name: str) -> None:
        if name not in self.img_dict:
            self.img_dict[name] = Image(
                self.px2mm,
                raw_img_path=name,
                all_methods_n_params=self.all_methods_n_params,
                methods_handler=self.methods_handler,
                bknd_img_path=self.bknd_img_path,
            )

    def step_1_main(self, index) -> None:
        name = self.img_path_list[index]
        self.initialize_image(name)

        self.img_dict[name].processing_image_before_filtering(self.algorithm)
        return self.img_dict[name].labels_on_img_before_filter

    def step_2_main(self, index) -> None:
        name = self.img_path_list[index]
        self.img_dict[name].load_filter_params(self.filter_param_dict)
        self.img_dict[name].initialize_circle_handler()
        self.img_dict[name].labels_filtering()
        self.img_dict[name].fill_ellipses()
        self.img_dict[name].overlay_ellipses_on_images()

        return self.img_dict[name].ellipses_on_images

    def ellipse_manual_adjustment(self, index) -> None:
        name = self.img_path_list[index]
        image = self.img_dict[name]
        self.adjuster = EllipseAdjuster(image.ellipses, image.img_rgb)

        loop = QEventLoop()

        def on_finished():
            self.handle_ellipse_adjustment_finished(image)
            loop.quit()

        self.adjuster.finished.connect(on_finished)
        self.adjuster.show()

        loop.exec()
        return image.ellipses_on_images

    def handle_ellipse_adjustment_finished(self, image: Image) -> None:
        print("ellipse handler triggered")
        print(image.filter_param_dict)
        image.update_ellipses(self.adjuster.ellipses)
        image.overlay_ellipses_on_images()
        print("ellipse handler finished ")

    def batch_process_images(
        self, worker_thread: WorkerThread, if_save: bool, save_path: Path = None
    ) -> None:
        # Process every image in the list
        for index, name in enumerate(self.img_path_list):
            print("-----------------")
            print("if_save", if_save)
            self.initialize_image(name)
            self.img_dict[name].load_filter_params(self.filter_param_dict)

            # Check if the image has been manually fine tuned
            if self.img_dict[name].if_fine_tuned:
                # If fine tuned, save the ellipses properties
                # and skip the processing
                print("This image has been fine tuned: ", name)
                self.img_dict[name].get_ellipse_properties()
                self.ellipses_properties.append(self.img_dict[name].ellipses_properties)

                if if_save:
                    self.save_processed_images(
                        self.img_dict[name].ellipses_on_images, name, save_path
                    )
                    continue

            self.img_dict[name].processing_image_before_filtering(self.algorithm)
            self.img_dict[name].filtering_processing()

            self.ellipses_properties.append(self.img_dict[name].ellipses_properties)

            if if_save:
                self.save_processed_images(
                    self.img_dict[name].ellipses_on_images, name, save_path
                )

            worker_thread.update_progress_bar(index + 1)
        worker_thread.on_processing_done()

    def save_processed_images(self, img, img_name, save_path) -> None:
        file_name = os.path.basename(img_name)
        new_name = os.path.join(save_path, file_name)
        print("new_saving_img_name:", new_name)
        try:
            cv2.imwrite(new_name, img)
            print("saved")
        except Exception as e:
            print(e)
