import pytest
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp_args():
    # Needed to ensure PyQt/PySide doesn't try to use standard platforms in headless CI
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    return []
