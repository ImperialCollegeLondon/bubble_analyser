
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg') 
import cv2
import numpy as np
import logging
from numpy import typing as npt
from bubble_analyser.processing.watershed_parent_class import WatershedSegmentation

class NormalWatershed(WatershedSegmentation):
    def __init__(self,
                 params: dict):
        
        self.name = "Normal Watershed"
        self.img_grey_dt_thresh: npt.NDArray[np.int_]
        self.sure_fg: npt.NDArray[np.int_]
        self.sure_bg: npt.NDArray[np.int_]
        self.unknown: npt.NDArray[np.int_]
        self.threshold_value: float
        self.resample: float
        self.update_params(params)
        
    def get_needed_params(self):
        return {
            "resample": self.resample,
            "element_size": self.element_size,
            "connectivity": self.connectivity,
            "threshold_value": self.threshold_value
        }
    
    def initialize_processing(self, 
                              params: dict,
                              img_grey: npt.NDArray[np.int_], 
                              img_rgb: npt.NDArray[np.int_],
                              bknd_img: npt.NDArray[np.int_] = None):
        
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.update_params(params)
        super().__init__(img_grey, 
                         img_rgb, 
                         bknd_img = bknd_img,
                         element_size=self.element_size, 
                         connectivity=self.connectivity)
        
    def update_params(self, params: dict):
        self.resample = params["resample"]
        self.element_size = params["element_size"]
        self.connectivity = params["connectivity"]
        self.threshold_value = params["threshold_value"]
        
    def __threshold_dt_image(self):
        _, self.img_grey_dt_thresh = cv2.threshold(
            self.img_grey_dt, self.threshold_value * self.img_grey_dt.max(), 255, cv2.THRESH_BINARY
        )
    
    def __get_sure_fg_bg(self):
        sure_fg_initial = self.img_grey_dt_thresh.copy()
        
        self.sure_bg = np.array(
            cv2.dilate(self.img_grey, np.ones((3, 3), np.uint8), iterations=1), dtype=np.uint8
        )
        self.sure_fg = np.array(sure_fg_initial, dtype=np.uint8)
        self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)
    
    def get_results_img(self):
        self._threshold()
        self._morph_process()
        self._dist_transform()
        self.__threshold_dt_image()
        self.__get_sure_fg_bg()
        self._initialize_labels(self.sure_fg)
        self._watershed_segmentation()
        self._overlay_labels_on_rgb()
        
        return self.labels_on_img, self.labels_watershed


class IterativeWatershed(WatershedSegmentation):
    def __init__(self, 
                 params: dict):

        self.name = "Iterative Watershed"
        self.max_thresh: float 
        self.min_thresh: float
        self.step_size: float
        self.update_params(params)
        self.output_mask_for_labels: npt.NDArray[np.int_]
        self.no_overlap_count: int = 0  # Track number of "no overlap" occurrences
        self.final_label_count: int = 0  # Track final number of labels

    def get_needed_params(self):
        return {
            "resample": self.resample,
            "element_size": self.element_size,
            "connectivity": self.connectivity,
            "max_thresh": self.max_thresh,
            "min_thresh": self.min_thresh,
            "step_size": self.step_size
        }
    
    def initialize_processing(self, 
                              params: dict,
                              img_grey: npt.NDArray[np.int_], 
                              img_rgb: npt.NDArray[np.int_],
                              bknd_img: npt.NDArray[np.int_] = None):
        
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.update_params(params)
        super().__init__(img_grey, 
                         img_rgb, 
                         bknd_img = bknd_img,
                         element_size=self.element_size, 
                         connectivity=self.connectivity)
        
    def update_params(self, params: dict):
        self.resample = params["resample"]
        self.element_size = params["element_size"]
        self.connectivity = params["connectivity"]
        self.max_thresh = params["max_thresh"]
        self.min_thresh = params["min_thresh"]
        self.step_size = params["step_size"]
        
    def __iterative_threshold(self):
        logging.basicConfig(level=logging.INFO)
        
        image = self.img_grey_dt.astype(np.uint8)
        
        # Initialize the final mask to accumulate all detected objects
        output_mask = np.zeros_like(image, dtype=np.uint8)

        # Set the initial threshold
        current_thresh = self.max_thresh
        self.no_overlap_count = 0  # Reset counter

        while current_thresh >= self.min_thresh:
            # Apply binary thresholding
            _, thresholded = cv2.threshold(image, current_thresh*image.max(), 255, cv2.THRESH_BINARY)
            
            # Label the thresholded image
            num_labels, labels = cv2.connectedComponents(thresholded, connectivity=self.connectivity)
            logging.basicConfig(level=logging.DEBUG)
            logging.info(f"Threshold {current_thresh:.2f}: {num_labels} components found.")

            # Detect new objects by comparing with the final mask
            for label in range(1, num_labels):  # Skip label 0 (background)
                # Create a mask for the current label
                component_mask = (labels == label).astype(np.uint8) * 255
                
                # component_max_intensity = cv2.minMaxLoc(image, mask=component_mask)[1]
                # Check if the object is already in the final mask
                overlap = cv2.bitwise_and(output_mask, component_mask)
                
                if not np.any(overlap):  # If no overlap, it's a new object

                    self.no_overlap_count += 1
                    output_mask = cv2.bitwise_or(output_mask, component_mask * 255)  

                    
            # Decrease the threshold for the next iteration
            current_thresh -= self.step_size
        self.output_mask_for_labels = output_mask
        
        self.final_label_count, _ = cv2.connectedComponents(self.output_mask_for_labels)
        logging.info(f"Total unique labels in output_mask_for_labels: {self.final_label_count}")
        logging.info(f"Total number of no overlap occurrences: {self.no_overlap_count}")
        
    def get_results_img(self):
        self._threshold()
        self._morph_process()
        self._dist_transform()
        self.__iterative_threshold()
        self._initialize_labels(self.output_mask_for_labels)
        self._watershed_segmentation()
        self._overlay_labels_on_rgb()
        return self.labels_on_img, self.labels_watershed
   

if __name__ == "__main__":
    
    # Define paths
    img_grey_path = "../../tests/test_image_grey.JPG"
    img_rgb_path = "../../tests/test_image_rgb.JPG"
    output_path = "../../tests/test_image_segmented.JPG"   # Change to your desired output location
    background_path = None  # Change if you have a background image

    # Load images
    img_rgb = cv2.imread(img_rgb_path)
    if img_rgb is None:
        raise ValueError(f"Error: Could not load image at {img_rgb_path}")

    img_grey = cv2.imread(img_grey_path, cv2.IMREAD_GRAYSCALE)
    
    # Load optional background image
    bknd_img = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE) if background_path else None

    # Run Iterative Watershed Segmentation
    iterative_watershed = IterativeWatershed(img_grey, img_rgb)
    
    segmented_img, labels_watershed = iterative_watershed.run_segmentation()
    pre_watershed_labels = iterative_watershed.output_mask_for_labels
    np.save("../../tests/test_labels_watershed.npy", labels_watershed)
    dist_transform = iterative_watershed.img_grey_dt
    
    # Save and display results
    plt.figure(figsize=(10, 5))
    plt.subplot(331)
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(332)
    plt.imshow(segmented_img, cmap="jet")
    plt.title("Segmented Image")

    plt.subplot(333)
    plt.imshow(pre_watershed_labels, cmap="gray")
    plt.title("Pre Watershed Labels")

    plt.subplot(334)
    plt.imshow(labels_watershed, cmap="jet")
    plt.title("Watershed Labels")

    plt.subplot(335)
    plt.imshow(dist_transform, cmap="gray")
    plt.title("Distance Transform")
    plt.savefig(output_path)
    plt.show()

    print(f"Segmentation completed! Output saved at: {output_path}")