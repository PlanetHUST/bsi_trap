"""Helpers for running repository scripts without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_src_to_path() -> Path:
    src_path = repo_root() / "src"
    src_text = str(src_path)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return src_path
