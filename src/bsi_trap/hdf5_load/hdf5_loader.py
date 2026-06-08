"""Backward-compatible import location."""

from .loader import HDF5Loader

hdf5_load_method = HDF5Loader

__all__ = ["HDF5Loader", "hdf5_load_method"]