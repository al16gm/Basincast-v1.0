"""Auditable single-series metrics. Undefined statistics return NaN."""
from __future__ import annotations
import numpy as np


def rmse(y_true, y_hat) -> float:
    a = np.asarray(y_true, dtype=float); b = np.asarray(y_hat, dtype=float)
    if len(a) == 0 or len(a) != len(b) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('nan')
    return float(np.sqrt(np.mean((a - b) ** 2)))


def kge(y_true, y_hat) -> float:
    a = np.asarray(y_true, dtype=float); b = np.asarray(y_hat, dtype=float)
    if len(a) < 3 or len(a) != len(b) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('nan')
    ma = float(a.mean()); mb = float(b.mean())
    sa = float(a.std(ddof=0)); sb = float(b.std(ddof=0))
    if abs(ma) < 1e-12 or sa < 1e-12 or sb < 1e-12:
        return float('nan')
    r = float(np.corrcoef(a, b)[0, 1]); alpha = sb/sa; beta = mb/ma
    return float(1.0 - np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2))


def nse(y_true, y_hat) -> float:
    a = np.asarray(y_true, dtype=float); b = np.asarray(y_hat, dtype=float)
    if len(a) < 2 or len(a) != len(b) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('nan')
    denom = float(np.sum((a-a.mean())**2))
    return float(1.0 - np.sum((a-b)**2)/denom) if denom > 1e-12 else float('nan')


def disjoint_origin_windows(origins, horizon: int) -> int:
    """Conservative diagnostic count of non-overlapping forecast intervals, NOT effective sample size."""
    if horizon < 1:
        raise ValueError('Horizon must be positive')
    chosen = 0
    next_allowed = -10**18
    for date in sorted(origins):
        period = int(date.year*12+date.month)
        if period >= next_allowed:
            chosen += 1
            next_allowed = period+horizon
    return chosen
