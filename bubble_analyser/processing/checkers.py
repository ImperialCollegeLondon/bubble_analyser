from config import Config
from pydantic import ValidationError
from PySide6.QtWidgets import QMessageBox
from bubble_analyser.gui import MainWindow

class ParamChecker:
    
    def __init__(self, params: Config, gui: MainWindow) -> None:
        self.gui: MainWindow = gui
        self.params: Config = params
        
    def param_checker(self, name: str, value: int|float) -> bool:
        
        if name == "element_size":
            try:
                self.params.element_size = value
            except ValidationError as e:
                self._show_warning("Invalid Element Size", str(e))
                return False
        
        if name == "connectivity":
            try:
                self.params.connectivity = value
            except ValidationError as e:
                self._show_warning("Invalid Connectivity", str(e))
                return False
        
        if name == "threshold_value":
            try:
                self.params.threshold_value = value
            except ValidationError as e:
                self._show_warning("Invalid Threshold Value", str(e))
                return False
        
        if name == "resample":
            try:
                self.params.resample = value
            except ValidationError as e:
                self._show_warning("Invalid Resample Factor", str(e))
                return False
        
        if name == "max_thresh":
            try:
                self.params.max_thresh = value
            except ValidationError as e:
                self._show_warning("Invalid Max Threshold", str(e))
                return False
        
        if name == "min_thresh":
            try:
                self.params.min_thresh = value
            except ValidationError as e:
                self._show_warning("Invalid Min Threshold", str(e))
                return False
        
        if name == "step_size":
            try:
                self.params.step_size = value
            except ValidationError as e:
                self._show_warning("Invalid Step Size", str(e))
                return False
        
        if name == "max_eccentricity":
            try:
                self.params.max_eccentricity = value
            except ValidationError as e:
                self._show_warning("Invalid Max Eccentricity", str(e))
                return False
        
        if name == "min_solidity":
            try:
                self.params.min_solidity = value
            except ValidationError as e:
                self._show_warning("Invalid Min Solidity", str(e))
                return False
        
        if name == "min_size":
            try:
                self.params.min_size = value
            except ValidationError as e:
                self._show_warning("Invalid Min Size", str(e))
                return False
        
        return True
    
    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self.gui, title, message)