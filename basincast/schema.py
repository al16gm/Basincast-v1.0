"""Strict, generic monthly series schema; no Jucar-specific dependencies."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd

ASSET_TYPES = {
    "river", "aquifer", "reservoir", "agricultural_demand", "urban_demand", "custom"
}
REQUIRED = {"series_id", "date", "value", "unit", "asset_type", "data_status", "source"}
ALLOWED_STATUS = {"OBSERVED_REPORTED", "USER_PROVIDED_UNVERIFIED", "RECONSTRUCTED", "ESTIMATED", "SYNTHETIC_DEMO"}
NONNEGATIVE = {"river", "reservoir", "agricultural_demand", "urban_demand"}

@dataclass
class SeriesData:
    series_id: str
    asset_type: str
    unit: str
    data_status: str
    source: str
    values: pd.Series
    source_file: str
    capacity: float | None = None

    @property
    def n_valid(self) -> int:
        return int(self.values.notna().sum())


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_series(path: str | Path) -> SeriesData:
    """Load one monthly CSV; never interpolate missing target observations."""
    path = Path(path)
    df = pd.read_csv(path, dtype={"series_id": "string", "asset_type": "string",
                                  "data_status": "string", "source": "string", "unit": "string"})
    _assert(REQUIRED.issubset(df.columns), f"Missing columns: {sorted(REQUIRED - set(df.columns))}")
    _assert(len(df) > 0, "Empty CSV")
    for name in ("series_id", "unit", "asset_type", "data_status", "source"):
        _assert(df[name].notna().all() and df[name].nunique(dropna=False) == 1,
                f"{name} must be constant, non-missing within one input series")
    asset = str(df.asset_type.iloc[0]); status = str(df.data_status.iloc[0])
    _assert(asset in ASSET_TYPES, f"Unsupported asset_type {asset!r}")
    _assert(status in ALLOWED_STATUS, f"Unsupported data_status {status!r}")
    raw_dates = pd.to_datetime(df.date, errors="raise")
    _assert(raw_dates.notna().all() and (raw_dates.dt.day == 1).all(),
            "All dates must be first day of month, e.g. 2019-01-01")
    _assert(not raw_dates.duplicated().any(), "Duplicate months are not allowed")
    numbers = pd.to_numeric(df.value, errors="coerce")
    bad_num = df.value.notna() & numbers.isna() & (df.value.astype(str).str.strip() != "")
    _assert(not bad_num.any(), "Invalid numeric value in observations")
    _assert(np.isfinite(numbers.dropna()).all(), "Non-finite numbers are not allowed")
    if asset in NONNEGATIVE:
        _assert(not (numbers.dropna() < 0).any(), "Negative values incompatible with asset type")
    capacity = None
    if "capacity" in df.columns and df.capacity.notna().any():
        caps = pd.to_numeric(df.capacity.dropna(), errors="raise")
        _assert((caps > 0).all() and caps.nunique() == 1, "capacity must be a single positive value")
        capacity = float(caps.iloc[0])
        if asset == "reservoir":
            _assert((numbers.dropna() <= capacity + 1e-9).all(), "Observed reservoir exceeds capacity")
    dates = pd.DatetimeIndex(raw_dates)
    vals = pd.Series(numbers.to_numpy(dtype=float), index=dates, dtype=float).sort_index()
    calendar = pd.date_range(vals.index.min(), vals.index.max(), freq="MS")
    vals = vals.reindex(calendar)
    _assert(int(vals.notna().sum()) > 0, "No valid target observations")
    _assert(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", str(df.series_id.iloc[0]))),
            "series_id must be a safe file identifier")
    return SeriesData(series_id=str(df.series_id.iloc[0]), asset_type=asset,
                      unit=str(df.unit.iloc[0]), data_status=status,
                      source=str(df.source.iloc[0]), values=vals,
                      source_file=path.name, capacity=capacity)
