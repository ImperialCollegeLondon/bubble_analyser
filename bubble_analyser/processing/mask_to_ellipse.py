"""Module for converting labeled/masked images back to ellipse format.

This module provides functionality to read labeled images (where each region has a unique label)
and convert them back into ellipse format compatible with the circle_handler module.
It performs the reverse operation of create_labelled_image_from_ellipses.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy import typing as npt
from skimage import measure


class MaskToEllipseConverter:
    """Converts labeled/masked images back to ellipse format.
    
    This class reads labeled images where each region has a unique integer label
    and fits ellipses to each labeled region, returning them in the same format
    used by the EllipseHandler class.
    """
    
    def __init__(self, min_contour_points: int = 5, validate_ellipses: bool = True):
        """Initialize the converter.
        
        Args:
            min_contour_points (int): Minimum number of contour points required to fit an ellipse.
                                    Default is 5 (minimum for cv2.fitEllipse).
            validate_ellipses (bool): Whether to validate ellipse parameters before including them.
                                    Default is True.
        """
        self.min_contour_points = min_contour_points
        self.validate_ellipses = validate_ellipses
        
    def load_labeled_image(self, image_path: str) -> npt.NDArray[np.int_]:
        """Load a labeled image from file.
        
        Args:
            image_path (str): Path to the labeled image file.
            
        Returns:
            npt.NDArray[np.int_]: The loaded labeled image.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be loaded.
        """
        try:
            # Try loading as numpy array first (for .npy files)
            if image_path.endswith('.npy'):
                labeled_img = np.load(image_path)
            else:
                # Load as regular image and convert to grayscale if needed
                labeled_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if labeled_img is None:
                    raise ValueError(f"Could not load image from {image_path}")
                
                # If it's a color image, convert to grayscale
                if len(labeled_img.shape) == 3:
                    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
                    
            return labeled_img.astype(np.int_)
            
        except Exception as e:
            raise ValueError(f"Error loading labeled image from {image_path}: {e}")
    
    def convert_mask_to_ellipses(
        self, 
        labeled_image: npt.NDArray[np.int_],
        background_label: int = 1
    ) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
        """Convert a labeled image to a list of ellipses.
        
        Args:
            labeled_image (npt.NDArray[np.int_]): Labeled image where each region has a unique label.
            background_label (int): Label value representing the background. Default is 1.
            
        Returns:
            list[tuple[tuple[float, float], tuple[float, float], float]]: List of ellipses in format
                ((center_x, center_y), (width, height), angle).
        """
        ellipses = []
        unique_labels = np.unique(labeled_image)
        
        logging.info(f"Found {len(unique_labels)} unique labels in the image")
        
        for label in unique_labels:
            # Skip background label
            if label == background_label or label == 0:
                continue
                
            # Create binary mask for current label
            mask = np.zeros_like(labeled_image, dtype=np.uint8)
            mask[labeled_image == label] = 255
            
            # Find contours for this label
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) >= self.min_contour_points:
                    try:
                        # Preprocess contour to ensure it's in the right format
                        contour = self._preprocess_contour(contour)
                        
                        # Additional check after preprocessing
                        if contour is None or len(contour) < self.min_contour_points:
                            logging.warning(f"Contour preprocessing failed for label {label}")
                            continue
                        
                        # Fit ellipse to contour
                        ellipse = cv2.fitEllipse(contour)
                        
                        if self.validate_ellipses:
                            if self._is_valid_ellipse(ellipse):
                                ellipses.append(ellipse)
                                logging.debug(f"Successfully fitted ellipse for label {label}")
                            else:
                                logging.warning(f"Invalid ellipse fitted for label {label}, skipping")
                        else:
                            ellipses.append(ellipse)
                            
                    except (cv2.error, ValueError, TypeError) as e:
                        logging.warning(f"Failed to fit ellipse for label {label}: {e}")
                        continue
                else:
                    logging.warning(f"Insufficient contour points ({len(contour)}) for label {label}")
        
        logging.info(f"Successfully converted {len(ellipses)} regions to ellipses")
        return ellipses
    
    def _preprocess_contour(self, contour: npt.NDArray) -> Optional[npt.NDArray]:
        """Preprocess contour to ensure it's in the correct format for cv2.fitEllipse.
        
        Args:
            contour: Input contour from cv2.findContours.
            
        Returns:
            Optional[npt.NDArray]: Preprocessed contour or None if preprocessing fails.
        """
        try:
            # Ensure contour is a numpy array
            if not isinstance(contour, np.ndarray):
                contour = np.array(contour)
            
            # First, extract just the x,y coordinates regardless of input format
            if contour.ndim == 3:
                if contour.shape[2] >= 2:
                    # Standard format: (n_points, 1, 2) or (n_points, 1, 3+)
                    # Take only first 2 columns (x, y coordinates)
                    points = contour[:, 0, :2]
                else:
                    logging.warning(f"Unexpected 3D contour shape: {contour.shape}")
                    return None
            elif contour.ndim == 2:
                if contour.shape[1] >= 2:
                    # Format: (n_points, 2) or (n_points, 3+)
                    # Take only first 2 columns (x, y coordinates)
                    points = contour[:, :2]
                else:
                    logging.warning(f"Unexpected 2D contour shape: {contour.shape}")
                    return None
            else:
                logging.warning(f"Unexpected contour dimensions: {contour.ndim}")
                return None
            
            # Ensure we have valid coordinate data
            if points.shape[1] != 2:
                logging.warning(f"Could not extract x,y coordinates from contour shape: {contour.shape}")
                return None
            
            # Convert to the format expected by cv2.fitEllipse: (n_points, 1, 2)
            contour = points.reshape(-1, 1, 2)
            
            # Ensure contour points are integers (required by OpenCV)
            contour = contour.astype(np.int32)
            
            # Remove duplicate consecutive points
            if len(contour) > 1:
                # Calculate differences between consecutive points
                diff = np.diff(contour.reshape(-1, 2), axis=0)
                # Find points that are different from the previous point
                non_duplicate_mask = np.any(diff != 0, axis=1)
                # Always keep the first point
                keep_mask = np.concatenate([[True], non_duplicate_mask])
                contour = contour[keep_mask]
            
            # Final validation
            if len(contour) < 5:  # cv2.fitEllipse needs at least 5 points
                return None
                
            return contour
            
        except Exception as e:
            logging.warning(f"Contour preprocessing error: {e}")
            return None
    
    def _is_valid_ellipse(self, ellipse: tuple[tuple[float, float], tuple[float, float], float]) -> bool:
        """Validate ellipse parameters.
        
        Args:
            ellipse: Ellipse in format ((center_x, center_y), (width, height), angle).
            
        Returns:
            bool: True if ellipse is valid, False otherwise.
        """
        center, axes, angle = ellipse
        center_x, center_y = center
        width, height = axes
        
        # Check if all parameters are finite
        if not all(np.isfinite([center_x, center_y, width, height, angle])):
            return False
            
        # Check if dimensions are positive
        if width <= 0 or height <= 0:
            return False
            
        return True
    
    def convert_file_to_ellipses(
        self, 
        image_path: str,
        background_label: int = 1
    ) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
        """Load a labeled image file and convert it to ellipses.
        
        Args:
            image_path (str): Path to the labeled image file.
            background_label (int): Label value representing the background. Default is 1.
            
        Returns:
            list[tuple[tuple[float, float], tuple[float, float], float]]: List of ellipses.
        """
        labeled_image = self.load_labeled_image(image_path)
        return self.convert_mask_to_ellipses(labeled_image, background_label)
    
    def get_ellipse_properties(
        self, 
        ellipses: list[tuple[tuple[float, float], tuple[float, float], float]]
    ) -> list[dict[str, float]]:
        """Calculate properties for each ellipse.
        
        Args:
            ellipses: List of ellipses in format ((center_x, center_y), (width, height), angle).
            
        Returns:
            list[dict[str, float]]: List of dictionaries containing ellipse properties.
        """
        properties = []
        
        for i, ellipse in enumerate(ellipses):
            center, axes, angle = ellipse
            width, height = axes
            
            # Calculate properties
            major_axis = max(width, height)
            minor_axis = min(width, height)
            area = np.pi * (major_axis / 2) * (minor_axis / 2)
            perimeter = np.pi * (
                3 * (major_axis + minor_axis)
                - np.sqrt((3 * major_axis + minor_axis) * (major_axis + 3 * minor_axis))
            )
            
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
                
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            properties.append({
                "ellipse_index": i,
                "area": area,
                "eccentricity": eccentricity,
                "equivalent_diameter": equivalent_diameter,
            })
            
        return properties
    
    def process_folder(
        self,
        folder_path: str,
        background_label: int = 1,
        save_results: bool = True,
        output_folder: Optional[str] = None
    ) -> dict[str, list[tuple[tuple[float, float], tuple[float, float], float]]]:
        """Process all files with '_mask.png' suffix in a folder.
        
        Args:
            folder_path (str): Path to the folder containing mask files.
            background_label (int): Label value representing the background. Default is 1.
            save_results (bool): Whether to save results to files. Default is True.
            output_folder (Optional[str]): Folder to save results. If None, saves to input folder.
            
        Returns:
            dict[str, list]: Dictionary mapping filename to list of ellipses.
            
        Raises:
            FileNotFoundError: If the folder doesn't exist.
            ValueError: If no mask files are found.
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Find all files with '_mask.png' suffix
        mask_files = list(folder_path.glob("*_mask.png"))
        
        if not mask_files:
            raise ValueError(f"No files with '_mask.png' suffix found in {folder_path}")
        
        logging.info(f"Found {len(mask_files)} mask files to process")
        
        results = {}
        
        # Set up output folder
        if save_results:
            if output_folder is None:
                output_folder = folder_path
            else:
                output_folder = Path(output_folder)
                output_folder.mkdir(parents=True, exist_ok=True)
        
        for mask_file in mask_files:
            try:
                logging.info(f"Processing {mask_file.name}")
                
                # Convert mask to ellipses
                ellipses = self.convert_file_to_ellipses(str(mask_file), background_label)
                results[mask_file.name] = ellipses
                
                logging.info(f"Successfully converted {len(ellipses)} regions from {mask_file.name}")
                
                # Save results if requested
                if save_results:
                    self._save_ellipse_results(mask_file, ellipses, output_folder)
                
            except Exception as e:
                logging.error(f"Error processing {mask_file.name}: {e}")
                results[mask_file.name] = []
        
        logging.info(f"Completed processing {len(mask_files)} files")
        return results
    
    def _save_ellipse_results(
        self,
        mask_file: Path,
        ellipses: list[tuple[tuple[float, float], tuple[float, float], float]],
        output_folder: Path
    ) -> None:
        """Save ellipse results to files.
        
        Args:
            mask_file (Path): Original mask file path.
            ellipses: List of ellipses.
            output_folder (Path): Output folder path.
        """
        try:
            base_name = mask_file.stem.replace("_mask", "")
            
            # Save ellipses as numpy array (convert to flat format)
            ellipses_file = output_folder / f"{base_name}_ellipses.npy"
            
            if ellipses:
                # Convert ellipses from ((center_x, center_y), (width, height), angle) 
                # to flat array [center_x, center_y, width, height, angle]
                ellipses_array = np.array([
                    [ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]]
                    for ellipse in ellipses
                ])
                np.save(ellipses_file, ellipses_array)
            else:
                # Save empty array for no ellipses
                np.save(ellipses_file, np.array([]))
            
            # Save properties as text file
            properties = self.get_ellipse_properties(ellipses)
            properties_file = output_folder / f"{base_name}_ellipse_properties.txt"
            
            with open(properties_file, 'w') as f:
                f.write(f"Ellipse properties for {mask_file.name}\n")
                f.write(f"Total ellipses: {len(ellipses)}\n\n")
                
                if properties:
                    for prop in properties:
                        f.write(f"Ellipse {prop['ellipse_index']}:\n")
                        f.write(f"  Equivalent Diameter: {prop['equivalent_diameter']:.1f}\n")
                        f.write(f"  Area: {prop['area']:.1f}\n")
                        f.write(f"  Eccentricity: {prop['eccentricity']:.3f}\n\n")
                else:
                    f.write("No ellipses found in this image.\n")
            
            logging.info(f"Saved results to {ellipses_file} and {properties_file}")
            
        except Exception as e:
            logging.error(f"Error saving results for {mask_file.name}: {e}")
            raise


def load_ellipses_from_mask(
    image_path: str,
    background_label: int = 1,
    min_contour_points: int = 5,
    validate_ellipses: bool = True
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Convenience function to load ellipses from a masked image file.
    
    Args:
        image_path (str): Path to the labeled image file.
        background_label (int): Label value representing the background. Default is 1.
        min_contour_points (int): Minimum contour points required for ellipse fitting.
        validate_ellipses (bool): Whether to validate ellipse parameters.
        
    Returns:
        list[tuple[tuple[float, float], tuple[float, float], float]]: List of ellipses.
    """
    converter = MaskToEllipseConverter(min_contour_points, validate_ellipses)
    return converter.convert_file_to_ellipses(image_path, background_label)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python mask_to_ellipse.py <path_to_labeled_image>")
        print("  Folder batch: python mask_to_ellipse.py <folder_path> --folder")
        print("  Folder batch with output: python mask_to_ellipse.py <folder_path> --folder --output <output_folder>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    is_folder_mode = "--folder" in sys.argv
    
    # Check for output folder option
    output_folder = None
    if "--output" in sys.argv:
        try:
            output_idx = sys.argv.index("--output")
            if output_idx + 1 < len(sys.argv):
                output_folder = sys.argv[output_idx + 1]
        except (ValueError, IndexError):
            print("Error: --output flag requires a folder path")
            sys.exit(1)
    
    try:
        converter = MaskToEllipseConverter()
        
        if is_folder_mode:
            # Process all mask files in folder
            print(f"Processing all '_mask.png' files in folder: {input_path}")
            if output_folder:
                print(f"Results will be saved to: {output_folder}")
            
            results = converter.process_folder(
                input_path, 
                save_results=True, 
                output_folder=output_folder
            )
            
            total_ellipses = sum(len(ellipses) for ellipses in results.values())
            print(f"\nBatch processing completed:")
            print(f"  Files processed: {len(results)}")
            print(f"  Total ellipses found: {total_ellipses}")
            
            # Summary of each file
            for filename, ellipses in results.items():
                print(f"  {filename}: {len(ellipses)} ellipses")
                
        else:
            # Process single file
            print(f"Processing single file: {input_path}")
            ellipses = converter.convert_file_to_ellipses(input_path)
            
            print(f"Successfully converted {len(ellipses)} regions to ellipses")
            
            # Calculate and display properties
            properties = converter.get_ellipse_properties(ellipses)
            
            for prop in properties:
                print(f"Ellipse {prop['ellipse_index']}:")
                print(f"  Equivalent Diameter: {prop['equivalent_diameter']:.1f}")
                print(f"  Area: {prop['area']:.1f}")
                print(f"  Eccentricity: {prop['eccentricity']:.3f}")
                print()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)