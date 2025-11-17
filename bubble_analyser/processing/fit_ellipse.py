"""Ellipse adjustment module for the Bubble Analyser application.

This module provides a graphical user interface for manually adjusting ellipses detected in images.
It allows users to fine-tune the position, size, and orientation of ellipses to improve accuracy
of bubble measurements when automatic detection is not perfect.
"""

import math
from typing import cast

import cv2
import numpy as np
from numpy import typing as npt
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EllipseAdjuster(QMainWindow):
    """A graphical user interface for manually adjusting ellipses on images.

    This class provides a window with controls for selecting, moving, resizing, and rotating
    ellipses that have been detected in an image. It allows for fine-tuning of automatically
    detected ellipses or adding new ellipses manually.

    The interface displays the image with ellipses overlaid and provides control points
    for manipulating the ellipses. The major and minor axes of each ellipse are shown
    with different colors, and control points are provided at the endpoints of each axis,
    at the center, and at rotation handles.

    Keyboard Controls:
    - 's' + 'x': Scale in major-axis direction (follows cursor)
    - 's' + 'y': Scale in minor-axis direction (follows cursor)
    - 'r': Rotate ellipse (follows cursor)
    - Space: Move ellipse (follows cursor)
    - Left mouse click: Exit current mode

    Attributes:
        finished (Signal): Signal emitted when the window is closed, passing the original
            image and the final list of adjusted ellipses.
        ellipses (list): List of ellipse tuples, where each tuple contains (center, axes, angle).
        B (ndarray): The original RGB image.
        dragging (bool): Flag indicating whether an axis endpoint is being dragged.
        moving_center (bool): Flag indicating whether an ellipse center is being moved.
        rotating (bool): Flag indicating whether an ellipse is being rotated.
        selected_ellipse (int): Index of the ellipse currently being manipulated.
        selected_ellipse_index (int): Index of the currently selected ellipse.
        selected_point (tuple): Last recorded mouse position in image coordinates.
        selected_axis (int): Index of the selected control point (0-3 for endpoints,
            4 for center, 5-6 for rotation handles).
        adding_new_circle (bool): Flag indicating whether a new circle is being added.
        scale_factor (float): Scaling factor for display.
        keyboard_mode (str): Current keyboard interaction mode ('scale_x', 'scale_y', 'rotate', 'move', None).
        s_key_pressed (bool): Flag indicating if 's' key is currently pressed.
    """

    # Signal to emit when the window is closed.
    # It sends the original image and the final ellipses list.
    finished = Signal(object, object)

    def __init__(
        self,
        ellipse_list: list[tuple[tuple[float, float], tuple[int, int], float]],
        img_rgb: npt.NDArray[np.int_],
    ) -> None:
        """Constructor for EllipseAdjuster.

        Parameters
        ----------
        ellipse_list : list of dicts
            List of dictionaries with ellipse parameters.
            Each dictionary should have the following keys:
                * x, y : float
                    The (x, y) coordinates of the ellipse's center.
                * a, b : float
                    The lengths of the ellipse's axes.
                * angle : float
                    The angle of the ellipse in degrees.
        img_rgb : numpy array
            The image to be displayed in the window.
        """
        super().__init__()
        self.setWindowTitle("Ellipse Adjuster")

        # Set window to fullscreen by default
        self.showMaximized()

        # Create main widget and layout
        main_widget = QWidget()
        self.main_layout = QHBoxLayout(main_widget)

        # Image display setup
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.image_label.setMouseTracking(True)
        self.main_layout.addWidget(self.image_label, 85)

        self.ellipses: list[tuple[tuple[float, float], tuple[int, int], float]] = ellipse_list.copy()
        self.B = img_rgb.copy()
        self.window_width = 1200
        self.dot_size = 3

        # Control panel setup
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # Delete button
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected_ellipse)
        control_layout.addWidget(self.delete_btn)

        # Add a circle button
        self.add_circle_btn = QPushButton("Add a circle")
        self.add_circle_btn.clicked.connect(self.add_circle_button_clicked)
        control_layout.addWidget(self.add_circle_btn)

        # Save and Exit button
        self.save_exit_btn = QPushButton("Save and Exit")
        self.save_exit_btn.clicked.connect(self.close)
        control_layout.addWidget(self.save_exit_btn)

        self.main_layout.addWidget(control_panel, 15)
        self.setCentralWidget(main_widget)

        # State variables for ellipse manipulation
        self.dragging = False
        self.moving_center = False  # For dragging the center of an ellipse.
        self.rotating = False  # For rotation via rotation handle.
        self.selected_ellipse: int = cast(int, None)  # Index of the ellipse being manipulated.
        self.selected_ellipse_index: int = cast(int, None)  # Used for highlighting/deletion.
        self.selected_point: tuple[float, float] = cast(
            tuple[float, float], None
        )  # Last recorded mouse position (in image coordinates).
        self.selected_axis: int = cast(int, None)  # 0-3 for endpoints, 4 for center, 5 for rotation handle.
        self.adding_new_circle = False  # Flag for adding a new circle.

        # New keyboard interaction variables
        self.keyboard_mode: str = cast(str, None)  # 'scale_x', 'scale_y', 'rotate', 'move', None
        self.s_key_pressed = False  # Track if 's' key is pressed
        self.keyboard_reference_point: tuple[float, float] = cast(
            tuple[float, float], None
        )  # Reference point for keyboard operations
        self.mode_active = False  # Track if any mode is currently active
        self.mode_reference_point: tuple[float, float] = cast(
            tuple[float, float], None
        )  # Reference point for mode operations
        self.mode_initial_values: dict = {}  # Store initial values for mode operations

        # New variables for drag-to-create functionality
        self.creating_new_circle = False  # Flag for drag-to-create mode
        self.new_circle_center: tuple[float, float] = cast(
            tuple[float, float], None
        )  # Center of the circle being created
        self.temp_circle_index: int = cast(int, None)  # Index of temporary circle

        self.scale_factor = 1.0

        # For rotation handling:
        self.initial_handle_angle = 0.0  # Angle between center and mouse when rotation started (radians).
        self.initial_ellipse_angle = 0.0  # Ellipse's original angle at start of rotation (degrees).
        self.default_axes = (100, 100)  # Both axes the same for a circle.
        self.default_angle = 0

        # Enable keyboard focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.update_image()

    def update_image(self) -> None:
        """Updates the displayed image with the current list of ellipses.

        Copies the stored image (self.B) and draws the ellipses on it.
        Then scales the image to a fixed width of 800 pixels while maintaining aspect ratio.
        Finally, converts the image to a QImage and displays it in the QLabel.
        """
        display_image = self.B.copy()
        self.draw_ellipses(display_image, self.ellipses)

        # Scale the image to a fixed width (800 pixels) while maintaining aspect ratio.
        height, width, _ = display_image.shape
        self.scale_factor = self.window_width / width
        new_size = (self.window_width, int(height * self.scale_factor))
        scaled_image = cv2.resize(display_image, new_size, interpolation=cv2.INTER_AREA)

        bytes_per_line = 3 * new_size[0]
        q_img = QImage(
            scaled_image.data,
            new_size[0],
            new_size[1],
            bytes_per_line,
            QImage.Format.Format_BGR888,
        )
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def draw_ellipses(
        self,
        image: npt.NDArray[np.int_],
        ellipses: list[tuple[tuple[float, float], tuple[int, int], float]],
    ) -> None:
        """Draws ellipses on the image.

        Parameters
        ----------
        image : np.ndarray
            The image to draw on.
        ellipses : list of tuple
            A list of tuples, where each tuple is (center, axes, angle).
            center is a tuple of (x, y) coordinates, axes is a tuple of (major, minor) axes lengths,
            and angle is the angle of the ellipse in degrees.
        """
        for idx, ellipse in enumerate(ellipses):
            # ellipse is a tuple: (center, axes, angle)
            center: tuple[float, float]
            axes: tuple[int, int]
            angle: float
            center, axes, angle = ellipse

            # Highlight selected ellipse in red; otherwise, use green.
            color = (0, 0, 255) if idx == self.selected_ellipse_index else (0, 255, 0)
            cv2.ellipse(image, ellipse, color, 2)  # type: ignore

            angle_rad = np.deg2rad(angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            # Compute endpoints for the major and minor axes.
            major_axis1 = (
                int(center[0] + axes[0] / 2 * cos_angle),
                int(center[1] + axes[0] / 2 * sin_angle),
            )
            major_axis2 = (
                int(center[0] - axes[0] / 2 * cos_angle),
                int(center[1] - axes[0] / 2 * sin_angle),
            )
            minor_axis1 = (
                int(center[0] + axes[1] / 2 * -sin_angle),
                int(center[1] + axes[1] / 2 * cos_angle),
            )
            minor_axis2 = (
                int(center[0] - axes[1] / 2 * -sin_angle),
                int(center[1] - axes[1] / 2 * cos_angle),
            )

            # Draw the axis lines.
            cv2.line(image, (int(center[0]), int(center[1])), major_axis1, (255, 0, 0), 2)
            cv2.line(image, (int(center[0]), int(center[1])), major_axis2, (255, 0, 0), 2)
            cv2.line(image, (int(center[0]), int(center[1])), minor_axis1, (0, 0, 255), 2)
            cv2.line(image, (int(center[0]), int(center[1])), minor_axis2, (0, 0, 255), 2)

            # Draw small circles at each endpoint.
            cv2.circle(image, major_axis1, self.dot_size, (0, 255, 255), -1)
            cv2.circle(image, major_axis2, self.dot_size, (0, 255, 255), -1)
            cv2.circle(image, minor_axis1, self.dot_size, (0, 255, 255), -1)
            cv2.circle(image, minor_axis2, self.dot_size, (0, 255, 255), -1)

            # Draw a cross at the center.
            center_int = (int(center[0]), int(center[1]))
            cross_size = 7
            cv2.line(
                image,
                (center_int[0] - cross_size, center_int[1]),
                (center_int[0] + cross_size, center_int[1]),
                (255, 255, 0),
                2,
            )
            cv2.line(
                image,
                (center_int[0], center_int[1] - cross_size),
                (center_int[0], center_int[1] + cross_size),
                (255, 255, 0),
                2,
            )

            # Compute and draw rotation handles.
            # We use two angles (45° and 225° in the ellipses local parametric space)
            for t_deg in (45, 225):
                t = np.deg2rad(t_deg)
                # Parametric form for an ellipse:
                # x = center[0] + (axes[0]/2)*cos(t)*cos(angle_rad) - (axes[1]/2)*sin(t)*sin(angle_rad)
                # y = center[1] + (axes[0]/2)*cos(t)*sin(angle_rad) + (axes[1]/2)*sin(t)*cos(angle_rad)
                rot_x = int(center[0] + (axes[0] / 2) * np.cos(t) * cos_angle - (axes[1] / 2) * np.sin(t) * sin_angle)
                rot_y = int(center[1] + (axes[0] / 2) * np.cos(t) * sin_angle + (axes[1] / 2) * np.sin(t) * cos_angle)
                # Draw a circle (e.g., magenta) for rotation handle.
                cv2.circle(image, (rot_x, rot_y), self.dot_size, (255, 0, 255), -1)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard events for Blender-like interaction modes."""
        key = event.key()

        # Handle 'A' key for adding new circle (no selection required)
        if key == Qt.Key.Key_A and not self.mode_active:
            self.add_circle_button_clicked()
            return

        # Only allow other keyboard modes when an ellipse is selected and no other mode is active
        if self.selected_ellipse_index is None:
            super().keyPressEvent(event)
            return

        # Handle 'X' key for major-axis scaling (simplified from s+x)
        if key == Qt.Key.Key_X and not self.mode_active:
            self.start_keyboard_mode("scale_x")
            return

        # Handle 'C' key for minor-axis scaling (simplified from s+y)
        if key == Qt.Key.Key_C and not self.mode_active:
            self.start_keyboard_mode("scale_y")
            return

        # Handle 'R' key for rotation
        if key == Qt.Key.Key_R and not self.mode_active:
            self.start_keyboard_mode("rotate")
            return

        # Handle space key for moving
        if key == Qt.Key.Key_Space and not self.mode_active:
            self.start_keyboard_mode("move")
            return

        super().keyPressEvent(event)

    def start_keyboard_mode(self, mode: str) -> None:
        """Start a keyboard interaction mode."""
        if self.selected_ellipse_index is None:
            return

        self.keyboard_mode = mode
        self.mode_active = True
        self.s_key_pressed = False  # Reset s key state

        # Get current mouse position as reference
        cursor_pos = self.mapFromGlobal(self.cursor().pos())
        label_pos = self.image_label.mapFromParent(cursor_pos)
        x = label_pos.x() / self.scale_factor
        y = label_pos.y() / self.scale_factor
        self.mode_reference_point = (x, y)

        # Store initial values for the selected ellipse
        center, axes, angle = self.ellipses[self.selected_ellipse_index]
        self.mode_initial_values = {"center": center, "axes": axes, "angle": angle, "mouse_pos": (x, y)}

    def exit_keyboard_mode(self) -> None:
        """Exit the current keyboard mode."""
        self.keyboard_mode = cast(str, None)
        self.mode_active = False
        self.s_key_pressed = False
        self.mode_reference_point = cast(tuple[float, float], None)
        self.mode_initial_values = {}

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard release events."""
        # No longer need to track 's' key release since we removed s+x/s+y combinations
        super().keyReleaseEvent(event)

    def get_current_mouse_position(self) -> tuple[float, float]:
        """Get current mouse position relative to the image."""
        cursor_pos = self.mapFromGlobal(self.cursor().pos())
        label_pos = self.image_label.mapFromParent(cursor_pos)
        x = label_pos.x() / self.scale_factor
        y = label_pos.y() / self.scale_factor
        return (x, y)

    def add_circle_button_clicked(self) -> None:
        """Called when the 'Add a circle' button is clicked."""
        self.adding_new_circle = True
        self.add_circle_btn.setText("Adding...")
        self.add_circle_btn.setStyleSheet("background-color: red; color: white;")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Called when the user presses a mouse button."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Left click exits keyboard modes
            if self.keyboard_mode is not None:
                self.exit_keyboard_mode()
                self.update_image()
                return

        if event.button() == Qt.MouseButton.RightButton:
            self.selected_ellipse_index = cast(int, None)
            self.selected_ellipse = cast(int, None)
            self.rotating = False
            self.moving_center = False
            self.dragging = False

            # Reset keyboard modes
            self.exit_keyboard_mode()

            # Cancel circle creation if in progress
            if self.creating_new_circle:
                self.creating_new_circle = False
                self.new_circle_center = cast(tuple[float, float], None)
                # Remove the temporary circle if it exists
                if self.temp_circle_index is not None and self.temp_circle_index < len(self.ellipses):
                    self.ellipses.pop(self.temp_circle_index)
                self.temp_circle_index = cast(int, None)

            self.update_image()
            return

        # Convert global mouse position to image_label coordinate system.
        label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
        x = label_pos.x() / self.scale_factor
        y = label_pos.y() / self.scale_factor

        # If we're in "adding new circle" mode, start drag-to-create.
        if self.adding_new_circle:
            self.creating_new_circle = True
            self.new_circle_center = (x, y)
            # Create a temporary circle with reasonable initial size
            new_ellipse = ((x, y), (20, 20), self.default_angle)  # Start with 20px radius
            self.ellipses.append(new_ellipse)
            self.temp_circle_index = len(self.ellipses) - 1
            self.selected_point = (x, y)
            self.update_image()
            return  # Skip further processing.

        # Otherwise, check if the click is near any control points.
        prev_selected = self.selected_ellipse_index
        self.dragging = False
        self.moving_center = False
        self.rotating = False  # Reset rotation state.

        for i, ellipse in enumerate(self.ellipses):
            # Skip the temporary circle being created
            if self.creating_new_circle and self.temp_circle_index is not None and i == self.temp_circle_index:
                continue

            center, axes, angle = ellipse
            angle_rad = np.deg2rad(angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            # Calculate endpoints for major and minor axes.
            major_axis1 = (
                int(center[0] + axes[0] / 2 * cos_angle),
                int(center[1] + axes[0] / 2 * sin_angle),
            )
            major_axis2 = (
                int(center[0] - axes[0] / 2 * cos_angle),
                int(center[1] - axes[0] / 2 * sin_angle),
            )
            minor_axis1 = (
                int(center[0] + axes[1] / 2 * -sin_angle),
                int(center[1] + axes[1] / 2 * cos_angle),
            )
            minor_axis2 = (
                int(center[0] - axes[1] / 2 * -sin_angle),
                int(center[1] - axes[1] / 2 * cos_angle),
            )

            # Calculate rotation handle positions (at 45° and 225°).
            rot_handles = []
            for t_deg in (45, 225):
                t = np.deg2rad(t_deg)
                rot_x = int(center[0] + (axes[0] / 2) * np.cos(t) * cos_angle - (axes[1] / 2) * np.sin(t) * sin_angle)
                rot_y = int(center[1] + (axes[0] / 2) * np.cos(t) * sin_angle + (axes[1] / 2) * np.sin(t) * cos_angle)
                rot_handles.append((rot_x, rot_y))

            # List of control points: endpoints then center, then rotation handles.
            control_points = [
                major_axis1,
                major_axis2,
                minor_axis1,
                minor_axis2,
                (int(center[0]), int(center[1])),
                *rot_handles,
            ]

            # Check each control point.
            for j, point in enumerate(control_points):
                if np.linalg.norm(np.array(point) - np.array((x, y))) < 20:
                    self.selected_ellipse_index = i
                    self.selected_ellipse = i
                    self.selected_point = (x, y)
                    self.selected_axis = j
                    if j == 4:
                        self.moving_center = True
                    elif j >= 5:
                        # Start rotation.
                        self.rotating = True
                        self.initial_handle_angle = math.atan2(y - center[1], x - center[0])
                        self.initial_ellipse_angle = cast(float, angle)
                    else:
                        self.dragging = True
                    break
            if self.dragging or self.moving_center or self.rotating:
                break

        if prev_selected != self.selected_ellipse_index:
            self.update_image()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handles mouse move events for adjusting ellipses."""
        label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
        x = label_pos.x() / self.scale_factor
        y = label_pos.y() / self.scale_factor

        # Handle keyboard-triggered modes
        if self.keyboard_mode is not None and self.selected_ellipse_index is not None and self.mode_initial_values:
            self.handle_keyboard_mode_movement(x, y)
            return

        # Handle drag-to-create circle - THIS MUST COME FIRST
        if self.creating_new_circle and self.new_circle_center is not None:
            center_x, center_y = self.new_circle_center
            # Calculate distance from center to current mouse position
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            # Use distance as diameter for both axes to create a circle
            diameter = max(10, int(distance * 2))  # Minimum diameter of 10px

            # Update the temporary circle
            if self.temp_circle_index is not None and self.temp_circle_index < len(self.ellipses):
                self.ellipses[self.temp_circle_index] = (
                    self.new_circle_center,
                    (diameter, diameter),  # Same diameter for both axes to make a circle
                    self.default_angle,
                )
            self.update_image()
            return

        # Existing ellipse adjustment logic
        if self.selected_ellipse is cast(int, None):
            return

        dx = x - self.selected_point[0]
        dy = y - self.selected_point[1]
        center, axes, angle = self.ellipses[self.selected_ellipse]
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        if self.moving_center:
            # Update the center of the ellipse.
            new_center = (center[0] + dx, center[1] + dy)
            self.ellipses[self.selected_ellipse] = (new_center, axes, angle)
        elif self.rotating:
            # Compute new handle angle and update the ellipse's angle.
            current_handle_angle = math.atan2(y - center[1], x - center[0])
            delta_angle = current_handle_angle - self.initial_handle_angle
            new_angle = self.initial_ellipse_angle + math.degrees(delta_angle)
            self.ellipses[self.selected_ellipse] = (center, axes, new_angle)
        elif self.dragging:
            # Adjust axis lengths based on which control point is dragged.
            if self.selected_axis == 0:  # Major axis endpoint 1
                new_axes = (
                    max(1, axes[0] + (dx * cos_angle + dy * sin_angle)),
                    axes[1],
                )
            elif self.selected_axis == 1:  # Major axis endpoint 2
                new_axes = (
                    max(1, axes[0] - (dx * cos_angle + dy * sin_angle)),
                    axes[1],
                )
            elif self.selected_axis == 2:  # Minor axis endpoint 1
                new_axes = (
                    axes[0],
                    max(1, axes[1] + (-dx * sin_angle + dy * cos_angle)),
                )
            elif self.selected_axis == 3:  # Minor axis endpoint 2
                new_axes = (
                    axes[0],
                    max(1, axes[1] - (-dx * sin_angle + dy * cos_angle)),
                )
            else:
                new_axes = axes
            self.ellipses[self.selected_ellipse] = (center, new_axes, angle)

        self.selected_point = (x, y)
        self.update_image()

    def handle_keyboard_mode_movement(self, x: float, y: float) -> None:
        """Handle mouse movement during keyboard modes."""
        if self.selected_ellipse_index is None or not self.mode_initial_values:
            return

        initial_center = self.mode_initial_values["center"]
        initial_axes = self.mode_initial_values["axes"]
        initial_angle = self.mode_initial_values["angle"]
        initial_mouse = self.mode_initial_values["mouse_pos"]

        if self.keyboard_mode == "move":
            # Move ellipse based on mouse movement
            dx = x - initial_mouse[0]
            dy = y - initial_mouse[1]
            new_center = (initial_center[0] + dx, initial_center[1] + dy)
            self.ellipses[self.selected_ellipse_index] = (new_center, initial_axes, initial_angle)

        elif self.keyboard_mode == "rotate":
            # Rotate based on mouse movement
            initial_angle_rad = math.atan2(initial_mouse[1] - initial_center[1], initial_mouse[0] - initial_center[0])
            current_angle_rad = math.atan2(y - initial_center[1], x - initial_center[0])
            angle_diff = math.degrees(current_angle_rad - initial_angle_rad)
            new_angle = (initial_angle + angle_diff) % 360
            self.ellipses[self.selected_ellipse_index] = (initial_center, initial_axes, new_angle)

        elif self.keyboard_mode == "scale_x":
            # Scale in major-axis direction based on distance from center
            distance_from_center = math.sqrt((x - initial_center[0]) ** 2 + (y - initial_center[1]) ** 2)
            initial_distance = math.sqrt(
                (initial_mouse[0] - initial_center[0]) ** 2 + (initial_mouse[1] - initial_center[1]) ** 2
            )
            if initial_distance > 0:
                scale_factor = distance_from_center / initial_distance
                new_major_axis = max(10, int(initial_axes[0] * scale_factor))
            else:
                new_major_axis = max(10, int(distance_from_center * 2))
            new_axes = (new_major_axis, initial_axes[1])
            self.ellipses[self.selected_ellipse_index] = (initial_center, new_axes, initial_angle)

        elif self.keyboard_mode == "scale_y":
            # Scale in minor-axis direction based on distance from center
            distance_from_center = math.sqrt((x - initial_center[0]) ** 2 + (y - initial_center[1]) ** 2)
            initial_distance = math.sqrt(
                (initial_mouse[0] - initial_center[0]) ** 2 + (initial_mouse[1] - initial_center[1]) ** 2
            )
            if initial_distance > 0:
                scale_factor = distance_from_center / initial_distance
                new_minor_axis = max(10, int(initial_axes[1] * scale_factor))
            else:
                new_minor_axis = max(10, int(distance_from_center * 2))
            new_axes = (initial_axes[0], new_minor_axis)
            self.ellipses[self.selected_ellipse_index] = (initial_center, new_axes, initial_angle)

        self.update_image()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handles mouse release events for adjusting ellipses."""
        # Handle completion of drag-to-create circle
        if self.creating_new_circle:
            self.creating_new_circle = False
            self.new_circle_center = cast(tuple[float, float], None)
            self.adding_new_circle = False
            self.temp_circle_index = cast(int, None)

            # Reset button appearance
            self.add_circle_btn.setText("Add a circle")
            self.add_circle_btn.setStyleSheet("")
            self.update_image()
            return

        # Don't reset states if in keyboard mode
        if self.keyboard_mode is not None:
            return

        # Existing release logic
        self.dragging = False
        self.moving_center = False
        self.rotating = False
        self.selected_ellipse = cast(int, None)
        self.selected_point = cast(tuple[float, float], None)
        self.selected_axis = cast(int, None)

    def delete_selected_ellipse(self) -> None:
        """Deletes the currently selected ellipse.

        This method removes the ellipse at the index specified by
        `selected_ellipse_index` from the list of ellipses, if an ellipse
        is selected. It then resets `selected_ellipse_index` to None and
        updates the displayed image to reflect the change.
        """
        if self.selected_ellipse_index is not None:
            del self.ellipses[self.selected_ellipse_index]  # type: ignore
            self.selected_ellipse_index = cast(int, None)
            self.update_image()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Overrides the default close event handler to generate a final image with ellipses overlaid.

        Before closing the window, this method creates a copy of the original image and overlays the
        adjusted ellipses on it. It then emits the finished signal with the original image and the
        list of adjusted ellipses. Optionally, it also writes the final image to a file.
        """
        display_image = self.B.copy()
        self.draw_ellipses(display_image, self.ellipses)

        # Emit the finished signal with the original image and the ellipses list.
        self.finished.emit(self.B, self.ellipses)
        event.accept()


# class Main(QMainWindow):
#     def __init__(self) -> None:
#         super().__init__()
#         self.setWindowTitle("Main Window")
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
#         layout = QVBoxLayout(central_widget)

#         # Button to launch the ellipse adjuster window.
#         self.launch_button = QPushButton("Generate Ellipse Adjuster")
#         self.launch_button.clicked.connect(self.openEllipseAdjuster)
#         layout.addWidget(self.launch_button)

#         # Label to display the final image with ellipses overlaid.
#         self.final_image_label = QLabel("Final Image with Ellipses will be shown here")
#         self.final_image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
#         layout.addWidget(self.final_image_label)

#     def openEllipseAdjuster(self) -> None:

#         self.adjuster = EllipseAdjuster(
#             np.load("../../tests/ellipses.npy", allow_pickle=True),
#             cv2.imread("../../tests/test_image_rgb.JPG")
#         )
#         self.adjuster.finished.connect(self.handleFinished)
#         self.adjuster.show()

#     def handleFinished(self, img, ellipses) -> None:
#         # Overlay the ellipses on a copy of the original image.
#         display_image = img.copy()
#         self.draw_ellipses(display_image, ellipses)

#         # Convert the image (BGR) to QImage for display.
#         height, width, _ = display_image.shape
#         bytes_per_line = 3 * width
#         q_img = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_BGR888)

#         self.final_image_label.setPixmap(QPixmap.fromImage(q_img))

#     def draw_ellipses(self, image, ellipses) -> None:
#         for idx, ellipse in enumerate(ellipses):
#             center, axes, angle = ellipse
#             # Highlight selected ellipse in red; otherwise, use green.
#             color = (0, 255, 0)
#             cv2.ellipse(image, ellipse, color, 2)

#             angle_rad = np.deg2rad(angle)
#             cos_angle = np.cos(angle_rad)
#             sin_angle = np.sin(angle_rad)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_window = Main()
#     main_window.show()
#     sys.exit(app.exec())
