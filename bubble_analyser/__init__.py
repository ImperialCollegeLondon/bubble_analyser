"""The main module for Bubble Analyser."""

# Try to get version from metadata, but fall back to a default if not available
try:
    from importlib.metadata import version

    __version__ = version(__name__)
except (ImportError, ModuleNotFoundError, Exception):
    # If importlib.metadata is not available or package is not found
    __version__ = "0.1.0"  # Default version when packaging with PyInstaller
