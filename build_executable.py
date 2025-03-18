"""Build script for creating a standalone executable of the Bubble Analyser application.

This script uses PyInstaller to package the Bubble Analyser application into a
standalone executable file. It handles the installation of PyInstaller if needed
and then runs the build process using the provided spec file.

Usage:
    python build_executable.py
"""

import subprocess
import sys


def check_pyinstaller() -> bool:
    """Check if PyInstaller is installed and install it if needed."""
    try:
        import PyInstaller  # noqa: F401

        print("PyInstaller is already installed.")
        return True
    except ImportError:
        print("PyInstaller is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("PyInstaller installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install PyInstaller. Please install it manually.")
            return False


def build_executable() -> None:
    """Build the executable using PyInstaller."""
    if not check_pyinstaller():
        return

    print("Building executable...")
    try:
        # Run PyInstaller with the spec file
        subprocess.check_call([sys.executable, "-m", "PyInstaller", "bubble_analyser.spec", "--clean"])
        print("\nBuild completed successfully!")
        print("The executable can be found in the 'dist' folder.")
    except subprocess.CalledProcessError as e:
        print(f"Error building executable: {e}")


if __name__ == "__main__":
    build_executable()
