"""The entry point for the Bubble Analyser program."""

from .default import main as default_main
from .GUI_manual import main as gui_main

if __name__ == "__main__":
    gui_main()
    # default_main()
    