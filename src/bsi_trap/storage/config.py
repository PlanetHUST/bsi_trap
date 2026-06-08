# bsi_trap/storage/config.py

from pathlib import Path
from .contract import StorageContract

DEFAULT_STORAGE = StorageContract(
    root_dir=Path("output"),
    final_dir_name="final_combined",
)
