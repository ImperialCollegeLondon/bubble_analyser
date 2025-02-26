from pathlib import Path

import toml as tomllib

from bubble_analyser.processing.config import Config

file_path = Path("config.toml")

params = Config(**tomllib.load(file_path))

params.element_size = 4
print(params)
