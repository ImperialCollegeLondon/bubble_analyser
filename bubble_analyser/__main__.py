"""The entry point for the Bubble Analyser program."""

from .default import main as default_main
from .GUI_manual import main as gui_main
from .test_2 import main as test_2_main
if __name__ == "__main__":
    # test_2_main()
    gui_main()
    # default_main()
    