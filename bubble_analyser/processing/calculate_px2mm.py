"""Module for calculating pixel to millimeter conversion ratios using calibration images.

This module provides functionality for calculating the conversion ratio between pixels
and millimeters in images. It includes an interactive QLabel subclass for measuring
distances in images and utility functions for image loading and processing.
"""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
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
    """A custom QLabel widget for interactive distance measurement in images.

    This class extends QLabel to allow users to draw lines on images by clicking and
    dragging. It's primarily used for measuring distances in calibration images to
    establish pixel-to-millimeter ratios.

    Attributes:
        refPt (list[tuple[int, int]]): List storing the start and end points of the
            measurement line.
        drawing (bool): Flag indicating whether the user is currently drawing a line.
        img (npt.NDArray[np.int_]): The original image being displayed.
        img_copy (npt.NDArray[np.int_]): A copy of the image for drawing operations.
        img_final (MatLike): The final image with the drawn line.
    """

    def __init__(self, parent: QWidget) -> None:
        """Initialize the ImageLabel widget.

        Args:
            parent (QWidget): The parent widget for this label.
        """
        super().__init__(parent)
        self.refPt: list[tuple[int, int]] = []
        self.drawing = False
        self.img: npt.NDArray[np.int_]
        self.img_copy: npt.NDArray[np.int_]
        self.img_final: MatLike

    def set_image(self, img: npt.NDArray[np.int_]) -> None:
        """Set the image to be displayed in the label.

        Args:
            img (npt.NDArray[np.int_]): The image array to display.
        """
        self.img = img
        self.img_copy = img.copy()
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(memoryview(img), width, height, bytes_per_line, QImage.Format.Format_RGB888)  # type: ignore
        self.setPixmap(QPixmap.fromImage(q_img))

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """Handle mouse press events for starting line drawing.

        Args:
            ev (QMouseEvent): The mouse event containing click coordinates.
        """
        if ev.button() == Qt.MouseButton.LeftButton:
            self.refPt.append((ev.x(), ev.y()))
            self.drawing = True

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        """Handle mouse move events for updating the line preview.

        Args:
            ev (QMouseEvent): The mouse event containing current coordinates.
        """
        if self.drawing:
            self.img_copy[:] = self.img[:]
            cv2.line(self.img_copy, self.refPt[0], (ev.x(), ev.y()), (0, 255, 0), 2)
            self.update_image()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        """Handle mouse release events for finalizing the drawn line.

        Args:
            ev (QMouseEvent): The mouse event containing final coordinates.
        """
        if ev.button() == Qt.MouseButton.LeftButton:
            self.refPt.append((ev.x(), ev.y()))
            self.drawing = False
            self.img_final = cv2.line(self.img, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            self.update_image()

    def update_image(self) -> None:
        """Update the displayed image with the current drawing state.

        Converts the numpy array image to a QImage and updates the label's pixmap.
        """
        height, width, channel = self.img_copy.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            memoryview(self.img_copy),  # type: ignore
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )  # type: ignore
        self.setPixmap(QPixmap.fromImage(q_img))


def resize_to_target_width(image: npt.NDArray[np.int_], target_width: int = 1000) -> tuple[npt.NDArray[np.int_], float]:
    """Resize an image to a target width while maintaining aspect ratio.

    Args:
        image (npt.NDArray[np.int_]): The input image to resize.
        target_width (int, optional): The desired width in pixels. Defaults to 1000.

    Returns:
        tuple[npt.NDArray[np.int_], float]: The resized image and the scale percentage.
    """
    scale_percent: float = target_width / image.shape[1]
    width: int = target_width
    height: int = int(image.shape[0] * scale_percent)
    dim: tuple[int, int] = (width, height)
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return cast(npt.NDArray[np.int_], image_resized), scale_percent


def get_pixel_distance(img: npt.NDArray[np.int_], main_window: QMainWindow) -> tuple[float, MatLike]:
    """Get the pixel distance between two points selected by the user.

    Opens a dialog window allowing the user to draw a line on the image and
    returns the length of that line in pixels.

    Args:
        img (npt.NDArray[np.int_]): The image to measure on.
        main_window (QMainWindow): The main window for displaying the dialog.

    Returns:
        float: The measured distance in pixels.
        Matlike: The image with the drawn line.
    """
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
    image_final = label.img_final
    if len(label.refPt) == 2:
        pixel_distance: float = np.sqrt(
            (label.refPt[1][0] - label.refPt[0][0]) ** 2 + (label.refPt[1][1] - label.refPt[0][1]) ** 2
        )
        return pixel_distance, image_final
    else:
        print("Line was not drawn correctly.")
        return 0.0, image_final


def get_mm_per_pixel(pixel_distance: float, scale_percent: float, img_resample: float) -> float:
    """Calculate millimeters per pixel based on measured pixel distance.

    Args:
        pixel_distance (float): The measured distance in pixels.
        scale_percent (float): The scale percentage used for image resizing.
        img_resample (float): Image resampling factor.

    Returns:
        float: The calculated millimeters per pixel ratio.
    """
    original_pixel_distance: float = pixel_distance / scale_percent
    mm_per_pixel: float = 10.0 / original_pixel_distance
    mm_per_pixel = mm_per_pixel / img_resample
    return mm_per_pixel


def calculate_px2mm(image_path: Path, img_resample: float, main_window: QMainWindow) -> tuple[float, float, MatLike]:
    """Calculate the pixel-to-millimeter conversion ratios for an image.

    This function allows the user to measure a known distance in an image and
    calculates both millimeters-per-pixel and pixels-per-millimeter ratios.

    Args:
        image_path (Path): Path to the image file.
        img_resample (float): Image resampling factor.
        main_window (QMainWindow): The main window for displaying the measurement dialog.

    Returns:
        tuple[float, float, Umat]: A tuple containing (millimeters per pixel, pixels per millimeter,
            ruler img with drawed line).
    """
    image = load_image(image_path)
    image, scale_percent = resize_to_target_width(image)
    pixel_distance: float
    img_final: MatLike
    pixel_distance, img_final = get_pixel_distance(image, main_window)

    mm_per_pixel = 0.0
    pixel_per_mm = 0.0

    if pixel_distance > 0:
        mm_per_pixel = get_mm_per_pixel(pixel_distance, scale_percent, img_resample)
        print(f"Conversion factor: {mm_per_pixel} mm per pixel")
        pixel_per_mm = 1 / mm_per_pixel
        print(f"Conversion factor: {pixel_per_mm} pixels per mm")

    return mm_per_pixel, pixel_per_mm, img_final
