"""The entry point for the Bubble Analyser program."""

import logging
import os
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog


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
        # Fallback to temp directory
        logs_dir = Path(tempfile.gettempdir()) / "BubbleAnalyser" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_str = now.strftime("%d%m%Y")
    timestamp_str = now.strftime("%H%M%S")
    log_filename = f"bubble_analyser_{date_str}_{timestamp_str}.log"
    log_file = logs_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


class DownloadWorker(QThread):
    """Thread to handle the 250MB weight download without freezing the GUI."""

    progress_changed = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, url, destination):
        super().__init__()
        self.url = url
        self.destination = destination

    def run(self):
        import requests

        try:
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(self.destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            self.progress_changed.emit(int((downloaded / total_size) * 100))
            self.finished.emit(True, "Success")
        except Exception as e:
            self.finished.emit(False, str(e))


def handle_weights_download(url, destination):
    """Shows a progress dialog for the weight download."""
    progress = QProgressDialog("Downloading ML Weights (mask_rcnn_bubble.h5)...", "Cancel", 0, 100)
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setWindowTitle("First Time Setup")

    worker = DownloadWorker(url, destination)
    worker.progress_changed.connect(progress.setValue)

    # Start thread and enter event loop
    worker.start()
    while worker.isRunning():
        QApplication.processEvents()
        if progress.wasCanceled():
            worker.terminate()
            return False
    return True


if __name__ == "__main__":
    # 1. Create the Application Singleton first
    app = QApplication.instance() or QApplication(sys.argv)

    try:
        setup_basic_logging()

        from bubble_analyser.gui.event_handlers import MainHandler
        from bubble_analyser.processing.error_handler import install_global_exception_handler
        from bubble_analyser.weights.loader import get_weights_path

        install_global_exception_handler()

        # 2. Check for weights
        w_path, w_url = get_weights_path(download_if_missing=False)

        if not w_path:
            # The weights are missing. Ask the user what to do.
            reply = QMessageBox.question(
                None,
                "Missing ML Weights",
                "The Machine Learning weights (mask_rcnn_bubble.h5, ~250 MB) are missing.\n\n"
                "These are required for CNN-based segmentation methods.\n"
                "Would you like to download them now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                # User accepted: Start the download with progress bar
                target_path = Path(__file__).parent / "weights" / "mask_rcnn_bubble.h5"
                logging.info(f"User accepted download. Target: {target_path}")

                success = handle_weights_download(w_url, target_path)

                if not success:
                    logging.warning("Download failed or cancelled during progress.")
            else:
                # User declined: Log it and proceed (CNN will be disabled in GUI)
                logging.info("User declined weight download. CNN methods will be unavailable.")

        # 3. Start the GUI
        # (The MainHandler will auto-detect the existing 'app' instance due to your previous fix)
        MainHandler()
        sys.exit(app.exec())

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logging.error(f"Error during application startup: {error_details}")
        print(f"Error during application startup: {e}\n{error_details}")
