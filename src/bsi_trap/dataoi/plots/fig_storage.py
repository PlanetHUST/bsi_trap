"""Figure output helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def ensure_dir(path: str | Path) -> Path:
    """Create and return an output directory."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def stem_with_suffix(stem: str, filename_suffix: str | None = None) -> str:
    """Append a filesystem-safe suffix to a base stem."""
    if not filename_suffix:
        return stem
    safe = "".join(
        char.lower() if char.isalnum() else "_" for char in str(filename_suffix)
    )
    safe = "_".join(safe.split("_")).strip("_")
    return f"{stem}_{safe}" if safe else stem


def save_figure(fig: Figure, outdir: str | Path, stem: str, dpi: int = 300) -> Path:
    """Save a figure as PDF and PNG, returning the PDF path."""
    out = ensure_dir(outdir)
    pdf_path = out / f"{stem}.pdf"
    png_path = out / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return pdf_path
