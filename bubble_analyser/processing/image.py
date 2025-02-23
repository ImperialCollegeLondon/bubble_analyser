from pathlib import Path
from .config import Config
from .image_preprocess import image_preprocess
from .calculate_px2mm import calculate_px2mm
from .threshold_methods import ThresholdMethods
import bubble_analyser.methods.watershed_methods
from ..methods.watershed_methods import IterativeWatershed, NormalWatershed
from .circle_handler import CircleHandler
import toml as tomllib
import importlib.util
import os
import inspect


import numpy as np
from numpy import typing as npt

class MethodsHandler():
    def __init__(self, params: Config) -> None:
        
        self.params_dict = params.model_dump()
        self.folder_path: Path = "/mnt/c/new_sizer/bubble_analyser/bubble_analyser/methods"\
    
        self.modules: dict[str, object] = {}
        self.modules = self.load_modules_from_folder()
        
        self.all_classes: dict[str, object] = {}

        self.full_dict: dict[str, dict[str, float|int]] = {}
        self._get_full_dict()

    def load_modules_from_folder(self, folder_path: str = None):
        if not folder_path:
            folder_path = self.folder_path
        modules = {}
        folder = Path(folder_path)
        
        for file in folder.glob("*.py"):
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[module_name] = module
        return modules

    def get_new_classes(self, module = None):
        if not module:
            module = self.modules
            
        new_classes = {}
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                new_classes[name] = obj
        return new_classes
    
    def _get_full_dict(self):

        for module_name, module in self.modules.items():
            print(f"Module: {module_name}")
            new_classes = self.get_new_classes(module)
            for class_name, class_obj in new_classes.items():
                print(f"  Class: {class_name}")
                
                
                # if class_name == "NormalWatershed":
                    # individual_param_dict = {}
                    
                instance: IterativeWatershed|NormalWatershed = class_obj(self.params_dict)
                self.all_classes[instance.name] = instance
                
                # name_list = instance.get_needed_params()
                # for name in name_list:
                #     individual_param_dict[name] = self.params_dict[name]
                
                self.full_dict[instance.name] = instance.get_needed_params()
                    
        print("full dict is", self.full_dict)
        print("all classes is", self.all_classes)

            
        

    
class Image():
    
    def __init__(self,
                #  param_dict: dict,
                 px2mm: float,
                 raw_img_path: Path,
                 all_methods_n_params: dict,
                 methods_handler: MethodsHandler,
                 bknd_img_path: Path = None) -> None:
        
        self.filter_param_dict: dict = {
            "max_eccentricity": 0.0,
            "min_solidity": 0.0,
            "min_size": 0.0
            }
        
        # self.segment_param_dict: dict = param_dict
        self.px2mm: float = px2mm
        self.raw_img_path = raw_img_path
        self.bknd_img_path: Path = None
        self.bknd_img: npt.NDArray[np.int_] = None
        
        if bknd_img_path is not None:
            self.if_bknd_img = True
            self.bknd_img_path = bknd_img_path
            
        self.img_rgb: npt.NDArray[np.int_]
        self.img_grey: npt.NDArray[np.int_]

        self.labels_on_img_before_filter: npt.NDArray[np.int_]
        self.labels_before_filter: npt.NDArray[np.int_]
        self.labels_after_filter: npt.NDArray[np.int_] = None
        self.ellipses: list = []
        self.ellipses_properties: list[dict[str, float]]
        self.ellipses_on_images: npt.NDArray[np.int_]
        
        self.all_methods_n_params: dict = all_methods_n_params
        
        self.methods_handler: MethodsHandler = methods_handler
        
        self.new_normal_watershed: NormalWatershed = None
        self.new_iterative_watershed: IterativeWatershed = None
        self.new_circle_handler: CircleHandler = None
        self.if_fine_tuned: bool = False
    
    def create_all_methods_instances(self):
        self.params_for_different_methods: dict[str, dict[str, float|int]] = {}
        for module_name, classes in self.methods_handler.all_classes.items():
            
            for class_name, class_obj in classes.items():
                if class_name == "NormalWatershed":
                    instance = class_obj(self.img_grey,
                                         self.img_rgb,
                                         self.segment_param_dict)
                    self.params_for_different_methods[instance.name] = instance.get_needed_params()
                    print(f"Created instance of {class_name} from module {module_name}")
                    
        print(self.params_for_different_methods)
    
    def return_params_for_method(self, method_name: str) -> list:
        return self.params_for_different_methods[method_name]

    def load_segment_params(self, dict_params: dict) -> None:
        pass 
        
    def load_filter_params(self, dict_params: dict) -> None:
        
        self.filter_param_dict = dict_params
        return
        
    def _img_preprocess(self, resample: float) -> None:
        
        # Get resized grey and RGB version of the target image
        self.img_grey, self.img_rgb = image_preprocess(self.raw_img_path, resample)
        if self.bknd_img_path is not None:
            self.bknd_img, _ = image_preprocess(self.bknd_img_path, resample)
        
        return
    
    def initialize_normal_watershed(self) -> None:

        self.new_normal_watershed = NormalWatershed(self.img_grey,
                                                    self.img_rgb,
                                                    bknd_img = self.bknd_img,
                                                    element_size = self.segment_param_dict["element_size"],
                                                    connectivity = self.segment_param_dict["connectivity"],
                                                    threshold_value = self.segment_param_dict["threshold_value"])

    def processing_image_before_filtering(self, algorithm: str) -> None:
        
        for algorithm_name, params in self.all_methods_n_params.items():
            
            print("algorithm name:", algorithm_name)
            
            if algorithm_name == algorithm:
                
                for name, processing_instance in self.methods_handler.all_classes.items():
                    
                    if name == algorithm_name:
                        
                        self._img_preprocess(params["resample"])
                        
                        processing_instance: IterativeWatershed|NormalWatershed
                        
                        instance = processing_instance.initialize_processing(params = params,
                                                                             img_grey = self.img_grey,
                                                                             img_rgb = self.img_rgb,
                                                                             bknd_img = self.bknd_img)
                
                        self.labels_on_img_before_filter, self.labels_before_filter = processing_instance.get_results_img()
                
                break
          
    def run_normal_watershed(self) -> None:  
        self.labels_on_img_before_filter, self.labels_before_filter = self.new_normal_watershed.run_segmentation()
        
    def initialize_iterative_watershed(self):

        self.new_iterative_watershed = IterativeWatershed(self.img_grey,
                                                          self.img_rgb,
                                                          bknd_img = self.bknd_img,
                                                          element_size = self.segment_param_dict["element_size"],
                                                          connectivity = self.segment_param_dict["connectivity"],
                                                          max_thresh = self.segment_param_dict["max_thresh"],
                                                          min_thresh = self.segment_param_dict["min_thresh"],
                                                          step_size = self.segment_param_dict["step_size"])
        return

    def run_iterative_watershed(self):
        self.labels_on_img_before_filter, self.labels_before_filter = self.new_iterative_watershed.run_segmentation()
    
    #--------------Below are filtering processes---------------------------
    def initialize_circle_handler(self):
        
        labels_before_filter = self.labels_before_filter.copy()
        rgb_img = self.img_rgb.copy()
        
        if self.new_circle_handler is not None:
            del self.new_circle_handler

        self.new_circle_handler = CircleHandler(labels_before_filter, 
                                                rgb_img, 
                                                self.px2mm)
        self.new_circle_handler.load_filter_params(self.filter_param_dict)
        return
    
    def labels_filtering(self):
        self.labels_after_filter = self.new_circle_handler.filter_labels_properties()
        return
    
    def fill_ellipses(self):
        self.ellipses = self.new_circle_handler.fill_ellipse_labels()
        return
    
    def update_ellipses(self, ellipses: list):
        # For manual modification of ellipses
        self.ellipses = ellipses
        self.new_circle_handler.ellipses = ellipses
        self.if_fine_tuned = True
        pass
        
    def get_ellipse_properties(self):
        self.ellipses_properties = self.new_circle_handler.calculate_circle_properties()
        return

    def overlay_ellipses_on_images(self):
        self.ellipses_on_images = self.new_circle_handler.overlay_ellipses_on_image()
        return
    
    def filtering_processing(self):
        self.initialize_circle_handler()
        self.labels_filtering()
        self.fill_ellipses()
        self.overlay_ellipses_on_images()
        self.get_ellipse_properties()
    
    
if __name__ == "__main__":
    a = Image()
    a.load_toml()
    print(a.img_resample)