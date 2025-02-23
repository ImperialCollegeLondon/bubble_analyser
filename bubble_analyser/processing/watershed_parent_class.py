import cv2
import numpy as np

from numpy import typing as npt
from bubble_analyser.processing.image_postprocess import overlay_labels_on_rgb
from bubble_analyser.processing.threshold_methods import ThresholdMethods
from bubble_analyser.processing.morphological_process import morphological_process

class WatershedSegmentation:
    def __init__(self, 
                 img_grey: npt.NDArray[np.int_], 
                 img_rgb: npt.NDArray[np.int_],
                 bknd_img: npt.NDArray[np.int_] = None,
                 element_size: int = 5,
                 connectivity: int = 4):
        
        self.img_grey: npt.NDArray[np.int_] = img_grey
        self.img_grey_thresholded: npt.NDArray[np.int_]
        self.img_grey_morph: npt.NDArray[np.int_]
        self.img_grey_dt: npt.NDArray[np.int_]
        self.img_rgb: npt.NDArray[np.int_] = img_rgb
        
        self.element_size: int = element_size
        self.connectivity: int = connectivity 
        self.labels: npt.NDArray[np.int_]
        self.labels_watershed: npt.NDArray[np.int_]
        self.labels_on_img: npt.NDArray[np.int_]

        self.bknd_img: npt.NDArray[np.int_] = bknd_img
                
    def _threshold(self):
        threshold_methods = ThresholdMethods()
        if self.bknd_img is not None:
            self.img_grey_thresholded = threshold_methods.threshold_with_background(self.img_grey, self.bknd_img)
        else:
            self.img_grey_thresholded = threshold_methods.threshold_without_background(self.img_grey)
    
    def _morph_process(self):

        self.img_grey_morph = morphological_process(self.img_grey_thresholded, self.element_size)
    
    def _dist_transform(self):
        
        self.img_grey_dt = cv2.distanceTransform(self.img_grey_morph, cv2.DIST_L2, self.element_size).astype(np.uint8)

    def _initialize_labels(self,
                            img: npt.NDArray[np.int_]):
        
        _, self.labels = cv2.connectedComponents(img, self.connectivity)
    
    def _watershed_segmentation(self):
        self.labels_watershed = cv2.watershed(self.img_rgb, self.labels).astype(np.int_)
    
    def _overlay_labels_on_rgb(self):
        self.labels_on_img = overlay_labels_on_rgb(
            self.img_rgb, self.labels_watershed
        )
    
