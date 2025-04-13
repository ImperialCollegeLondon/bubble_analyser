"""The entry point for the Bubble Analyser program."""

import logging
import sys
import traceback
from pathlib import Path


# Set up basic logging in case an error occurs before the main logging is configured
def setup_basic_logging() -> None:
    """Set up basic logging to ensure errors are captured before main logging is configured."""
    # Use user's home directory or temporary directory for logs
    import os
    import tempfile
    
    # Try to use the user's application data directory first
    user_home = Path.home()
    app_data_dir = user_home / "Library" / "Application Support" / "BubbleAnalyser"
    
    try:
        app_data_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = app_data_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
    except OSError:
        # Fallback to temp directory if we can't write to app data dir
        logs_dir = Path(tempfile.gettempdir()) / "BubbleAnalyser" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / "bubble_analyser.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


if __name__ == "__main__":
    try:
        setup_basic_logging()
        from bubble_analyser.gui.event_handlers import MainHandler
        from bubble_analyser.processing.error_handler import install_global_exception_handler

        # Install the global exception handler
        install_global_exception_handler()

        # Start the application
        MainHandler()
    except Exception as e:
        # Catch any exceptions during startup
        error_details = traceback.format_exc()
        logging.error(f"Error during application startup: {error_details}")

        # If GUI is not available yet, print to console
        print(f"Error during application startup: {e}\n{error_details}")

        # Try to show a basic error dialog if possible
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv)
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.setWindowTitle("Startup Error")
            error_box.setText(f"Error during application startup: {e}")
            error_box.setDetailedText(error_details)
            error_box.exec()
        except Exception:
            # If we can't show a dialog, at least the error is logged and printed
            pass
