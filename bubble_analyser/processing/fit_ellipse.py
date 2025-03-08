import math

import cv2
import numpy as np
from numpy import typing as npt
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EllipseAdjuster(QMainWindow):
    # Signal to emit when the window is closed.
    # It sends the original image and the final ellipses list.
    finished = Signal(object, object)

    def __init__(
        self,
        ellipse_list: list[tuple[tuple[float, float], tuple[int, int], int]],
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

        # Create main widget and layout
        main_widget = QWidget()
        self.main_layout = QHBoxLayout(main_widget)

        # Image display setup
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.image_label.setMouseTracking(True)
        self.main_layout.addWidget(self.image_label, 85)

        self.ellipses: list[tuple[tuple[float, float], tuple[int, int], int]] = ellipse_list.copy()
        self.B = img_rgb.copy()

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
        self.selected_ellipse = None  # Index of the ellipse being manipulated.
        self.selected_ellipse_index = None  # Used for highlighting/deletion.
        self.selected_point = None  # Last recorded mouse position (in image coordinates).
        self.selected_axis = None  # 0-3 for endpoints, 4 for center, 5 for rotation handle.
        self.adding_new_circle = False  # Flag for adding a new circle.
        self.scale_factor = 1.0

        # For rotation handling:
        self.initial_handle_angle = 0.0  # Angle between center and mouse when rotation started (radians).
        self.initial_ellipse_angle = 0  # Ellipse's original angle at start of rotation (degrees).

        # Default new ellipse parameters (a circle)
        self.default_axes = (100, 100)  # Both axes the same for a circle.
        self.default_angle = 0

        self.update_image()

    def update_image(self) -> None:
        """Updates the displayed image with the current list of ellipses.
        Copies the stored image (self.B) and draws the ellipses on it.
        Then scales the image to a fixed width of 1500 pixels while maintaining aspect ratio.
        Finally, converts the image to a QImage and displays it in the QLabel.
        """
        display_image = self.B.copy()
        self.draw_ellipses(display_image, self.ellipses)

        # Scale the image to a fixed width (1500 pixels) while maintaining aspect ratio.
        height, width, _ = display_image.shape
        self.scale_factor = 1500 / width
        new_size = (1500, int(height * self.scale_factor))
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
        ellipses: list[tuple[tuple[float, float], tuple[int, int], int]],
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
            cv2.circle(image, major_axis1, 8, (0, 255, 255), -1)
            cv2.circle(image, major_axis2, 8, (0, 255, 255), -1)
            cv2.circle(image, minor_axis1, 8, (0, 255, 255), -1)
            cv2.circle(image, minor_axis2, 8, (0, 255, 255), -1)

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
            # We use two angles (45° and 225° in the ellipse’s local parametric space)
            for t_deg in (45, 225):
                t = np.deg2rad(t_deg)
                # Parametric form for an ellipse:
                # x = center[0] + (axes[0]/2)*cos(t)*cos(angle_rad) - (axes[1]/2)*sin(t)*sin(angle_rad)
                # y = center[1] + (axes[0]/2)*cos(t)*sin(angle_rad) + (axes[1]/2)*sin(t)*cos(angle_rad)
                rot_x = int(center[0] + (axes[0] / 2) * np.cos(t) * cos_angle - (axes[1] / 2) * np.sin(t) * sin_angle)
                rot_y = int(center[1] + (axes[0] / 2) * np.cos(t) * sin_angle + (axes[1] / 2) * np.sin(t) * cos_angle)
                # Draw a circle (e.g., magenta) for rotation handle.
                cv2.circle(image, (rot_x, rot_y), 8, (255, 0, 255), -1)

    def add_circle_button_clicked(self) -> None:
        """Called when the 'Add a circle' button is clicked."""
        self.adding_new_circle = True
        self.add_circle_btn.setText("Click the position where you want a new ellipse")
        self.add_circle_btn.setStyleSheet("background-color: red; color: white;")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Called when the user presses a mouse button.

        If the right button is clicked, any selection is cancelled.

        If the left button is clicked, it checks if the click is near any control points.
        If so, it sets `selected_ellipse_index` and `selected_ellipse` to the index of the ellipse,
        and `selected_point` to the coordinates of the click.
        If the click is on an endpoint, it sets `dragging` to True.
        If the click is on the center, it sets `moving_center` to True.
        If the click is on a rotation handle, it sets `rotating` to True and stores
        the initial handle angle and ellipse angle.
        If any of the above is true, it calls `update_image` to redraw the image with the new selection.
        """
        if event.button() == Qt.MouseButton.RightButton:
            self.selected_ellipse_index = None
            self.selected_ellipse = None
            self.rotating = False
            self.moving_center = False
            self.dragging = False
            self.update_image()
            return

        # Convert global mouse position to image_label coordinate system.
        label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
        x = label_pos.x() / self.scale_factor
        y = label_pos.y() / self.scale_factor

        # If we're in "adding new circle" mode, create a new ellipse.
        if self.adding_new_circle:
            new_ellipse = ((x, y), self.default_axes, self.default_angle)
            self.ellipses.append(new_ellipse)
            self.adding_new_circle = False
            self.add_circle_btn.setText("Add a circle")
            self.add_circle_btn.setStyleSheet("")
            self.update_image()
            return  # Skip further processing.

        # Otherwise, check if the click is near any control points.
        prev_selected = self.selected_ellipse_index
        self.dragging = False
        self.moving_center = False
        self.rotating = False  # Reset rotation state.

        for i, ellipse in enumerate(self.ellipses):
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
            ] + rot_handles

            # Check each control point.
            for j, point in enumerate(control_points):
                if np.linalg.norm(np.array(point) - np.array((x, y))) < 20:
                    self.selected_ellipse_index = i  # type: ignore
                    self.selected_ellipse = i  # type: ignore
                    self.selected_point = (x, y)  # type: ignore
                    # For indices:
                    # 0-3: endpoints, 4: center, 5-6: rotation handles.
                    self.selected_axis = j  # type: ignore
                    if j == 4:
                        self.moving_center = True
                    elif j >= 5:
                        # Start rotation.
                        self.rotating = True
                        # Store the initial handle angle and ellipse angle.
                        # Angle from center to click in radians.
                        self.initial_handle_angle = math.atan2(y - center[1], x - center[0])
                        self.initial_ellipse_angle = angle  # in degrees
                    else:
                        self.dragging = True
                    break
            if self.dragging or self.moving_center or self.rotating:
                break

        if prev_selected != self.selected_ellipse_index:
            self.update_image()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handles mouse move events for adjusting ellipses.

        This method is responsible for updating the position, rotation, or axis lengths of
        an ellipse based on mouse movement. The adjustments are made to the currently
        selected ellipse, if any.

        Parameters
        ----------
        event : QMouseEvent
            The event object containing information about the mouse move event.

        Behavior
        --------
        - If no ellipse is selected, the method returns immediately.
        - Calculates the change in mouse position relative to the previously recorded
        point of interaction.
        - If an ellipse center is being moved, updates the ellipse center.
        - If the ellipse is being rotated, calculates the new angle and updates the ellipse
        angle.
        - If an axis is being dragged, adjusts the axis lengths based on the control point
        being dragged.
        - Updates the image to reflect changes.
        """
        if self.selected_ellipse is None:
            return

        label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())  # type: ignore
        x = label_pos.x() / self.scale_factor
        y = label_pos.y() / self.scale_factor

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
            # current angle from center to mouse:
            current_handle_angle = math.atan2(y - center[1], x - center[0])
            # Change in angle (in radians)
            delta_angle = current_handle_angle - self.initial_handle_angle
            # Update ellipse angle (in degrees)
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handles mouse release events for adjusting ellipses.

        This method is responsible for resetting the state of the EllipseAdjuster to
        indicate that no ellipse is currently being edited.

        Parameters
        ----------
        event : QMouseEvent
            The event object containing information about the mouse release event.

        Behavior
        --------
        - Resets the ellipse adjustment state (dragging, moving center, rotating)
        - Resets the selected ellipse, point, and axis to None.
        """
        self.dragging = False
        self.moving_center = False
        self.rotating = False
        self.selected_ellipse = None
        self.selected_point = None
        self.selected_axis = None

    def delete_selected_ellipse(self) -> None:
        """Deletes the currently selected ellipse.

        This method removes the ellipse at the index specified by
        `selected_ellipse_index` from the list of ellipses, if an ellipse
        is selected. It then resets `selected_ellipse_index` to None and
        updates the displayed image to reflect the change.
        """
        if self.selected_ellipse_index is not None:
            del self.ellipses[self.selected_ellipse_index]  # type: ignore
            self.selected_ellipse_index = None
            self.update_image()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Overrides the default close event handler to generate a final image with ellipses overlaid.

        Before closing the window, this method creates a copy of the original image and overlays the
        adjusted ellipses on it. It then emits the finished signal with the original image and the
        list of adjusted ellipses. Optionally, it also writes the final image to a file.
        """
        display_image = self.B.copy()
        self.draw_ellipses(display_image, self.ellipses)

        # Optionally, you can write the file.
        cv2.imwrite("adjusted_ellipses.jpg", display_image)

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
