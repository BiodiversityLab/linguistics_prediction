from pathlib import Path
import sys
import os
import importlib


def load_module(path: Path):
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(module_dir)
    module = importlib.import_module(path)
    return module