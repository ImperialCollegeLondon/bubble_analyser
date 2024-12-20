# Bubble Analyser

[Description for project.]

This is a Python application that uses [poetry](https://python-poetry.org) for packaging
and dependency management. It also provides [pre-commit](https://pre-commit.com/) hooks
(for [ruff](https://pypi.org/project/ruff/) and
[mypy](https://mypy.readthedocs.io/en/stable/)) and automated tests using
[pytest](https://pytest.org/) and [GitHub Actions](https://github.com/features/actions).

## For developers

This is a Python application that uses [poetry](https://python-poetry.org) for packaging
and dependency management. It also provides [pre-commit](https://pre-commit.com/) hooks
for various linters and formatters and automated tests using
[pytest](https://pytest.org/) and [GitHub Actions](https://github.com/features/actions).
Pre-commit hooks are automatically kept updated with a dedicated GitHub Action.

To get started:

1. [Download and install Poetry](https://python-poetry.org/docs/#installation) following the instructions for your OS.
1. Clone this repository and make it your working directory
1. Set up the virtual environment:

   ```bash
   poetry install
   ```

1. Activate the virtual environment (alternatively, ensure any Python-related command is preceded by `poetry run`):

   ```bash
   poetry shell
   ```

1. Install the git hooks:

   ```bash
   pre-commit install
   ```

1. Run the main app:

   ```bash
   python -m bubble_analyser
   ```
