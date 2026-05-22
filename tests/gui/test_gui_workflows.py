import pytest
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from bubble_analyser.gui.event_handlers import MainHandler

@pytest.fixture
def app_handler(qtbot):
    """Fixture to initialize the main application handler and its GUI."""
    # Ensure there's a QApplication instance
    if QApplication.instance() is None:
        app = QApplication([])
    
    # Initialize the MainHandler
    handler = MainHandler()
    
    # Add the main window to qtbot so it cleans up correctly
    qtbot.addWidget(handler.gui)
    
    return handler

def test_initial_tab_state(app_handler):
    """Verify that the application starts on the Folder Selection tab."""
    gui = app_handler.gui
    # The initial tab should be index 0 (folder_selection_tab)
    assert gui.tabs.currentIndex() == 0

def test_algorithm_list_population(app_handler):
    """Verify that algorithms are populated in the combo box on the processing tab."""
    gui = app_handler.gui
    
    # The algorithm combo box should have items loaded
    assert gui.algorithm_combo.count() > 0
    
    # Verify that the default "Default" algorithm is present
    algorithms = [gui.algorithm_combo.itemText(i) for i in range(gui.algorithm_combo.count())]
    assert "Default" in algorithms

def test_parameter_update_sync(app_handler, qtbot):
    """Verify that changing a parameter in the GUI updates the backend model."""
    gui = app_handler.gui
    model = app_handler.image_processing_model
    
    # Select Iterative Watershed to test max_thresh
    index = gui.algorithm_combo.findText("Iterative Watershed")
    if index >= 0:
        gui.algorithm_combo.setCurrentIndex(index)
        
    # Find a parameter row, e.g., 'max_thresh' in param_sandbox1
    param_table = gui.param_sandbox1
    found_row = -1
    for row in range(param_table.rowCount()):
        if param_table.item(row, 0) and param_table.item(row, 0).text() == "max_thresh":
            found_row = row
            break
            
    assert found_row != -1, "max_thresh parameter not found in table"
    
    # Modify the value in the table
    new_value = "0.99"
    param_table.item(found_row, 1).setText(new_value)
    
    # Trigger the update (simulating clicking "Confirm Parameter Before Filtering" button)
    qtbot.mouseClick(gui.preview_button1, Qt.MouseButton.LeftButton)
    
    # Verify the model has the updated parameter
    assert model.all_methods_n_params["Iterative Watershed"]["max_thresh"] == 0.99
