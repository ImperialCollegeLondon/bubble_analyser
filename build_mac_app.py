import os
import shutil
import subprocess
import sys
from pathlib import Path

def build_app():
    """
    Build the macOS application using PyInstaller.
    """
    # Define paths
    project_root = Path(__file__).parent
    entry_point = project_root / "bubble_analyser" / "__main__.py"
    app_name = "Bubble Analyser"
    
    # Clean up previous builds
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # PyInstaller arguments
    args = [
        str(entry_point),
        "--name", app_name,
        "--noconfirm",
        "--windowed",  # Run as a GUI app (no terminal)
        "--clean",
        
        # Data files: source:dest
        # Include config.toml
        f"--add-data={project_root}/bubble_analyser/config.toml:bubble_analyser",
        # Include weights
        f"--add-data={project_root}/bubble_analyser/weights/mask_rcnn_bubble.h5:bubble_analyser/weights",
        # Include mrcnn
        f"--add-data={project_root}/bubble_analyser/mrcnn:bubble_analyser/mrcnn",
        # Include bubble
        f"--add-data={project_root}/bubble_analyser/bubble:bubble_analyser/bubble",
        
        # Hidden imports (if necessary, usually PyInstaller finds them but sometimes pandas/scipy need help)
        "--hidden-import=scipy.special.cython_special",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors.typedefs",
        "--hidden-import=sklearn.neighbors.quad_tree",
        "--hidden-import=sklearn.tree._utils",
        
        # Exclude unnecessary modules to save space (optional)
        # "--exclude-module=tkinter",
    ]

    # Run PyInstaller
    print(f"Building {app_name}...")
    subprocess.check_call(["pyinstaller"] + args)
    
    print(f"Build complete. App is located at: {dist_dir / (app_name + '.app')}")

if __name__ == "__main__":
    build_app()
