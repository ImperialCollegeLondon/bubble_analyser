"""The entry point for the Bubble Analyser program."""

import logging
import os
import platform
import tempfile
from pathlib import Path


def setup_basic_logging() -> None:
    """Set up cross-platform, user-writable logging before the main app runs."""
    try:
        system = platform.system()
        if system == "Windows":
            app_data_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "BubbleAnalyser"
        elif system == "Darwin":
            app_data_dir = Path.home() / "Library" / "Application Support" / "BubbleAnalyser"
        else:  # Linux or other Unix-like
            app_data_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "BubbleAnalyser"

        logs_dir = app_data_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback to temporary directory
        logs_dir = Path(tempfile.gettempdir()) / "BubbleAnalyser" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "bubble_analyser.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
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
        import traceback

        error_details = traceback.format_exc()
        logging.error(f"Error during application startup: {error_details}")

        # If GUI is not available yet, print to console
        print(f"Error during application startup: {e}\n{error_details}")

        # Try to show a basic error dialog if possible
        try:
            import sys

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
