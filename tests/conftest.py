import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CACHE = {}


def load_module(file_name, alias):
    if alias in _CACHE:
        return _CACHE[alias]
    spec = importlib.util.spec_from_file_location(alias, str(ROOT / file_name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    _CACHE[alias] = mod
    return mod


@pytest.fixture(scope="session")
def extractor():
    return load_module("1_data_extractor.py", "onyx_extractor")


@pytest.fixture(scope="session")
def demo():
    return load_module("3_demo_pipeline.py", "onyx_demo")


@pytest.fixture(scope="session")
def ml_pipeline():
    return load_module("2_ml_pipeline.py", "onyx_ml")

