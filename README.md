# Bubble Analyser

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-blue)](https://python-poetry.org/)

## Overview

Bubble Analyser is a Python application for detecting, measuring, and analyzing bubbles in images. It provides advanced image processing capabilities using watershed segmentation algorithms to accurately identify and measure bubbles of various sizes and distributions.

## Features

- **Image Processing**: Preprocess images with various transformations
- **Bubble Detection**: Identify bubbles using advanced watershed segmentation algorithms
- **Measurement**: Calculate size, shape, and distribution of bubbles
- **Calibration**: Convert pixel measurements to real-world units (mm)
- **Manual Adjustment**: Fine-tune detected bubbles with interactive tools
- **Results Visualization**: Generate histograms and statistics of bubble distributions
- **User-Friendly GUI**: Intuitive interface for different functionalities

## Installation

### Using the Executable (Windows)

For Windows users, a standalone executable is available in the release:

v0.1.0 <https://github.com/ImperialCollegeLondon/bubble_analyser/releases/tag/v0.1.0>

### From Source

To install from source, follow the developer setup instructions below.

## Usage

1. **Select Input Images**: Load images containing bubbles for analysis
2. **Calibrate**: Set up pixel-to-mm conversion using a reference image
3. **Process Images**: Apply detection algorithms to identify bubbles
4. **Adjust Results**: Fine-tune the detected bubbles if needed
5. **Export Data**: Save measurements and visualizations

## Methods

Bubble Analyser implements multiple watershed segmentation approaches:

- **Normal Watershed**: Standard watershed algorithm with a single threshold
- **Iterative Watershed**: Advanced algorithm that iteratively applies thresholds to detect objects at different intensity levels

## For Developers

This is a Python application that uses [poetry](https://python-poetry.org) for packaging
and dependency management. It also provides [pre-commit](https://pre-commit.com/) hooks
for various linters and formatters and automated tests using
[pytest](https://pytest.org/) and [GitHub Actions](https://github.com/features/actions).
Pre-commit hooks are automatically kept updated with a dedicated GitHub Action.

To get started:

1. [Download and install Poetry](https://python-poetry.org/docs/#installation) following the instructions for your OS.
2. Clone this repository and make it your working directory
3. Set up the virtual environment:

   ```bash
   poetry install
   ```

4. Activate the virtual environment:

   ```bash
   poetry shell
   ```

   alternatively, ensure any Python-related command is preceded by `poetry run'.

5. Install the git hooks:

   ```bash
   pre-commit install
   ```

6. Run the main app:

   ```bash
   python -m bubble_analyser
   ```

## Dependencies

- Python 3.12
- PySide6 (Qt for Python)
- NumPy
- SciPy
- scikit-image
- OpenCV
- Matplotlib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. If you would like to add more image processing algorithms, please see the guidance under the directory [Methods](bubble_analyser/methods)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Yiyang Guan - Department of Earth Science and Engineering, Imperial College London
- Diego Mesa Pena - Department of Earth Science and Engineering, Imperial College London
- Diego Alonso Álvarez - Imperial College London RSE Team, Imperial College London
