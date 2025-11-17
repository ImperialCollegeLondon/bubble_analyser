#!/usr/bin/env python3
"""
Simple test script to verify BubMask method can be loaded by MethodsHandler.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def test_bubmask_import():
    """Test if BubMaskWatershed can be imported and instantiated."""
    try:
        from bubmask_method import BubMaskWatershed
        print("✓ Successfully imported BubMaskWatershed")
        
        # Test instantiation with minimal parameters
        test_params = {
            "weights_path": "/path/to/weights.h5",
            "confidence_threshold": 0.9,
            "target_width": 800,
            "image_min_dim": 192,
            "image_max_dim": 384,
        }
        
        instance = BubMaskWatershed(test_params)
        print("✓ Successfully instantiated BubMaskWatershed")
        print(f"  - Name: {instance.name}")
        
        # Test get_needed_params
        params = instance.get_needed_params()
        print("✓ Successfully called get_needed_params()")
        print(f"  - Required params: {list(params.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_methods_handler():
    """Test if MethodsHandler can load BubMask method."""
    try:
        from processing.image import MethodsHandler
        from processing.config import Config
        
        # Create a minimal config
        config = Config()
        
        # Create MethodsHandler
        handler = MethodsHandler(config)
        print("✓ Successfully created MethodsHandler")
        
        # Check if BubMask method is loaded
        print(f"Loaded modules: {list(handler.modules.keys())}")
        print(f"Available methods: {list(handler.all_classes.keys())}")
        
        if "BubMask (Deep Learning)" in handler.all_classes:
            print("✓ BubMask method successfully detected by MethodsHandler!")
            return True
        else:
            print("✗ BubMask method NOT detected by MethodsHandler")
            return False
            
    except Exception as e:
        print(f"✗ Error with MethodsHandler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing BubMask method loading...")
    print("=" * 50)
    
    print("\n1. Testing direct import:")
    success1 = test_bubmask_import()
    
    print("\n2. Testing MethodsHandler:")
    success2 = test_methods_handler()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ All tests passed! BubMask method should be working.")
    else:
        print("✗ Some tests failed. Check the errors above.")