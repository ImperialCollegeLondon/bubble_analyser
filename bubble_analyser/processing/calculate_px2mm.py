from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from .image_preprocess import load_image


class ImageLabel(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.refPt: list[tuple[int, int]] = []
        self.drawing = False
        self.img: npt.NDArray[np.int_]
        self.img_copy: npt.NDArray[np.int_]

    def set_image(self, img: npt.NDArray[np.int_]) -> None:
        self.img = img
        self.img_copy = img.copy()
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            memoryview(img), width, height, bytes_per_line, QImage.Format.Format_RGB888
        )  # type: ignore
        self.setPixmap(QPixmap.fromImage(q_img))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.refPt.append((event.x(), event.y()))
            self.drawing = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drawing:
            self.img_copy[:] = self.img[:]
            cv2.line(
                self.img_copy, self.refPt[0], (event.x(), event.y()), (0, 255, 0), 2
            )
            self.update_image()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.refPt.append((event.x(), event.y()))
            self.drawing = False
            cv2.line(self.img, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            self.update_image()

    def update_image(self) -> None:
        height, width, channel = self.img_copy.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            memoryview(self.img_copy),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )  # type: ignore
        self.setPixmap(QPixmap.fromImage(q_img))


def resize_to_target_width(
    image: npt.NDArray[np.int_], target_width: int = 1000
) -> tuple[npt.NDArray[np.int_], float]:
    scale_percent: float = target_width / image.shape[1]
    width: int = target_width
    height: int = int(image.shape[0] * scale_percent)
    dim: tuple[int, int] = (width, height)
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return cast(npt.NDArray[np.int_], image_resized), scale_percent


def get_pixel_distance(img: npt.NDArray[np.int_], main_window: QMainWindow) -> float:
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Draw Line for Measurement")
    dialog.setModal(True)
    dialog.setGeometry(100, 100, img.shape[1], img.shape[0])

    label = ImageLabel(dialog)
    label.set_image(img)  # type: ignore
    layout = QVBoxLayout(dialog)
    layout.addWidget(label)
    dialog.setLayout(layout)
    dialog.show()

    # Wait for the user to draw the line
    while len(label.refPt) < 2:
        QApplication.processEvents()

    dialog.close()

    if len(label.refPt) == 2:
        pixel_distance: float = np.sqrt(
            (label.refPt[1][0] - label.refPt[0][0]) ** 2
            + (label.refPt[1][1] - label.refPt[0][1]) ** 2
        )
        return pixel_distance
    else:
        print("Line was not drawn correctly.")
        return 0.0


def get_mm_per_pixel(
    pixel_distance: float, scale_percent: float, img_resample: float
) -> float:
    original_pixel_distance: float = pixel_distance / scale_percent
    mm_per_pixel: float = 10.0 / original_pixel_distance
    mm_per_pixel = mm_per_pixel / img_resample
    return mm_per_pixel


def calculate_px2mm(
    image_path: Path, img_resample: float, main_window: QMainWindow
) -> tuple[float, float]:
    image = load_image(image_path)
    image, scale_percent = resize_to_target_width(image)
    pixel_distance: float = get_pixel_distance(image, main_window)

    mm_per_pixel = 0.0
    pixel_per_mm = 0.0

    if pixel_distance > 0:
        mm_per_pixel = get_mm_per_pixel(pixel_distance, scale_percent, img_resample)
        print(f"Conversion factor: {mm_per_pixel} mm per pixel")
        pixel_per_mm = 1 / mm_per_pixel
        print(f"Conversion factor: {pixel_per_mm} pixels per mm")

    return mm_per_pixel, pixel_per_mm
