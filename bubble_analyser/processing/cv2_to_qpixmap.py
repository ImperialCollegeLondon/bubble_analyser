"""Utility module for converting OpenCV images to Qt QPixmap objects.

This module provides functionality to convert OpenCV image matrices (numpy arrays)
to PySide6 QPixmap objects for display in Qt-based GUI applications. It handles
the necessary color space conversion from BGR to RGB and proper memory management
for seamless integration between OpenCV and Qt frameworks.
"""

from cv2.typing import MatLike
from PySide6.QtGui import QImage, QPixmap


def cv2_to_qpixmap(cv_img: MatLike) -> QPixmap:
    """Convert an OpenCV image (NumPy array) to a QPixmap.

    Args:
        cv_img (numpy.ndarray): OpenCV image in BGR format (NumPy array)

    Returns:
        PySide6.QtGui.QPixmap: The converted image as a QPixmap

    Raises:
        ValueError: If the input image is not a valid OpenCV image
    """
    import cv2

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # Get dimensions
    height, width, channels = rgb_image.shape

    # Create QImage from RGB data
    bytes_per_line = channels * width
    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(q_image)

    return pixmap
