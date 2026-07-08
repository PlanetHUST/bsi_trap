import os


def require_pagn_bits():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import pagn.constants as ct
        from pagn.opacities import electron_scattering_opacity
    except ImportError as exc:
        raise ImportError(
            "pAGN is required for migration torques. Install with `pip install pagn`."
        ) from exc
    return ct, electron_scattering_opacity


def constant(ct, *names: str) -> float:
    for name in names:
        if hasattr(ct, name):
            return getattr(ct, name)
    raise AttributeError(f"pagn.constants is missing one of: {', '.join(names)}")
