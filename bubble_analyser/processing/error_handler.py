"""Error handling utilities for the Bubble Analyser application.

This module provides functionality for capturing and displaying errors in a user-friendly way,
ensuring that errors are both logged to file and displayed to the user through the GUI.
"""

import logging
import sys
import traceback
from typing import Callable, Optional, Type

from PySide6.QtWidgets import QMessageBox


def show_error_dialog(title: str, message: str, details: Optional[str] = None) -> None:
    """Display an error message in a dialog box.

    Args:
        title: The title of the error dialog.
        message: The main error message to display.
        details: Optional detailed error information (e.g., traceback).
    """
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Icon.Critical)
    error_box.setWindowTitle(title)
    error_box.setText(message)
    
    if details:
        error_box.setDetailedText(details)
    
    error_box.exec()


def exception_handler(exctype: Type[BaseException], value: BaseException, tb) -> None: # type: ignore
    """Global exception handler to catch unhandled exceptions.

    This function logs the exception and displays an error dialog to the user.

    Args:
        exctype: The exception type.
        value: The exception value.
        tb: The traceback object.
    """
    # Format the traceback
    traceback_details = ''.join(traceback.format_exception(exctype, value, tb))
    
    # Log the exception
    logging.error(f"Unhandled exception: {traceback_details}")
    
    # Show error dialog
    error_message = f"An unexpected error occurred: {str(value)}"
    show_error_dialog("Application Error", error_message, traceback_details)


def install_global_exception_handler() -> None:
    """Install the global exception handler.

    This function should be called at the start of the application to ensure
    all unhandled exceptions are properly caught and displayed.
    """
    sys.excepthook = exception_handler