
import sys
import os
import time
import logging
import unittest
from pathlib import Path

# Add cnn_methods and project root to path
# Add cnn_methods and project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
cnn_methods_dir = project_root / "bubble_analyser"
sys.path.insert(0, str(cnn_methods_dir))
print(f"Project root: {project_root}")
print(f"CNN methods dir: {cnn_methods_dir}")
print(f"Exists: {cnn_methods_dir.exists()}")
print(f"Sys path: {sys.path[:3]}")

# import bubmask_wrapper
from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskDetector, BubMaskConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestParallelImplementation(unittest.TestCase):
    
    def setUp(self):
        self.weights_path = project_root / "bubble_analyser/weights/mask_rcnn_bubble.h5"
        self.input_dir = project_root / "tests/sample_images"
        self.output_dir = project_root / "tests/parallel_test_results"
        
        if not self.input_dir.exists():
            self.skipTest(f"Input directory not found: {self.input_dir}")
            
        if not self.weights_path.exists():
            self.skipTest(f"Weights file not found: {self.weights_path}")

        # Initialize detector
        print("\nInitializing detector...")
        config = BubMaskConfig.for_medium_quality()
        self.detector = BubMaskDetector(str(self.weights_path), config)

    def test_parallel_vs_sequential(self):
        print("\n=== Testing Parallel Implementation on macOS ===")
        
        # Test 1: Sequential Processing (Baseline) - explicitly calling the loop logic if possible
        # Since batch_detect now calls batch_detect_parallel, we can't easily test the OLD sequential method
        # unless we manually implement the loop here or force batch_size=1 (which is still "parallel" logic but sequential execution)
        
        print("\n--- Test 1: Sequential Processing (Batch Size = 1) ---")
        start_time = time.time()
        results_seq = self.detector.batch_detect_parallel(
            self.input_dir, 
            output_dir=self.output_dir / "sequential",
            save_masks=False,
            save_splash=False,
            batch_size=1
        )
        seq_duration = time.time() - start_time
        print(f"Sequential (Batch=1) processing took {seq_duration:.2f} seconds")
        print(f"Processed {len(results_seq)} images")
        
        # Test 2: Batch Parallel Processing
        print("\n--- Test 2: Batch Parallel Processing (Batch Size = 4) ---")
        start_time = time.time()
        results_par = self.detector.batch_detect_parallel(
            self.input_dir, 
            output_dir=self.output_dir / "parallel",
            save_masks=False,
            save_splash=False,
            batch_size=4
        )
        par_duration = time.time() - start_time
        print(f"Parallel (Batch=4) processing took {par_duration:.2f} seconds")
        print(f"Processed {len(results_par)} images")
        
        # Comparison
        print("\n--- Comparison ---")
        print(f"Sequential (Batch=1): {seq_duration:.2f}s")
        print(f"Parallel (Batch=4):   {par_duration:.2f}s")
        if par_duration < seq_duration:
            speedup = seq_duration / par_duration
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("No speedup observed (expected on small datasets or initialization overhead)")
            
        # Verify results match count
        self.assertEqual(len(results_seq), len(results_par), "Number of results mismatch!")
        print("SUCCESS: Number of results match.")

if __name__ == "__main__":
    unittest.main()
