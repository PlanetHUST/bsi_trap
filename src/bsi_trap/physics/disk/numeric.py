from types import SimpleNamespace

import numpy as np


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    num, den = np.broadcast_arrays(numerator, denominator)
    out = np.zeros_like(num, dtype=float)
    return np.divide(num, den, out=out, where=den != 0.0)


def sorted_disk_view(disk) -> SimpleNamespace:
    radius = np.asarray(disk.R, dtype=float)
    order = np.argsort(radius)
    data = {"Mbh": disk.Mbh, "Rs": disk.Rs}
    for name in ["R", "rho", "T", "h", "cs", "Omega", "tauV", "kappa", "Teff4"]:
        arr = np.asarray(getattr(disk, name), dtype=float)
        if arr.shape == radius.shape:
            arr = arr[order]
        data[name] = arr
    return SimpleNamespace(**data)


def radial_log_slope(radius, values, *, smoothing: float = 0.005, k: int = 3):
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(radius <= 0.0) or np.any(values <= 0.0):
        raise ValueError("radius and values must be positive for log-slope")

    radius_log = np.log10(radius)
    values_log = np.log10(values)

    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError as exc:
        raise ImportError(
            "scipy is required for smoothed torque-profile slopes. Install scipy "
            "or install pAGN with its scientific Python dependencies."
        ) from exc

    spline_order = min(int(k), radius_log.size - 1)
    spline = UnivariateSpline(
        radius_log, values_log, k=spline_order, s=float(smoothing), ext=0
    )
    return spline.derivative()(radius_log)
