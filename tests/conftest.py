"""Shared test configuration.

``@pytest.mark.slow`` marks tests that load real models (embedders, rerankers). They are
skipped unless ``--run-slow`` is given or ``MDK_RUN_SLOW=1`` is set.
"""
import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests marked slow (they load real models)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads real models; skipped unless --run-slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow") or os.environ.get("MDK_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: pass --run-slow or set MDK_RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
