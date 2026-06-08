"""Helpers for finding nodes inside HDF5 groups."""

from __future__ import annotations

import h5py


def get_node(root: h5py.Group | h5py.File, path: str) -> h5py.Group | h5py.Dataset | None:
    """Return ``root[path]`` if it exists; otherwise return ``None``."""
    normalized = path.strip("/")
    if not normalized:
        return root

    current: h5py.Group | h5py.Dataset = root
    for part in normalized.split("/"):
        if not isinstance(current, h5py.Group) or part not in current:
            return None
        current = current[part]

    return current


def require_group(root: h5py.Group | h5py.File, path: str) -> h5py.Group | None:
    node = get_node(root, path)
    return node if isinstance(node, h5py.Group) else None
