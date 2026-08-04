"""Shared test fixtures.

Cache files (`cache_*.json`) live at the repo root and `discover._cache_path`
resolves them relative to CWD, so anchor every test at the repo root regardless
of where pytest was invoked from.
"""
from __future__ import annotations

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
