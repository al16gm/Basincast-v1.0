"""Convert a worldwide two-column monthly CSV/XLSX into the BasinCast contract.

The user, not the software, selects the measured quantity, unit, provenance,
and aggregation operator. Missing monthly observations are never invented.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .schema import ASSET_TYPES, ALLOWED_STATUS, load_series


def normalize_user_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    series_id: str,
    asset_type: str,
    unit: str,
    date_column: str = "date",
    value_column: str = "value",
    data_status: str = "USER_PROVIDED_UNVERIFIED",
    source_description: str = "USER_UPLOAD_UNVERIFIED",
    aggregation: str | None = None,
    decimal_comma: bool = False,
    sheet_name: str | int = 0,
    capacity: float | None = None,
) -> dict:
    """Normalize monthly data; reject ambiguous multiple samples per month.

    The application must specify --aggregation mean/sum/last for daily or
    multiple same-month values. Choice depends on the physical quantity.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if asset_type not in ASSET_TYPES:
        raise ValueError(f"Unsupported asset_type {asset_type!r}")
    if data_status not in ALLOWED_STATUS:
        raise ValueError(f"Unsupported data_status {data_status!r}")
    if not unit.strip() or not source_description.strip():
        raise ValueError("Unit and source_description cannot be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", series_id):
        raise ValueError("series_id must contain only letters, numbers, _, - or .")
    if capacity is not None and (not np.isfinite(capacity) or capacity <= 0):
        raise ValueError("capacity must be a positive finite number")
    if aggregation not in {None, "mean", "sum", "last"}:
        raise ValueError("aggregation must be mean, sum, last or unset")
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        # Always allow comma-separated standard CSV; for semicolon CSV, detect
        # structural headers rather than guessing from locale-dependent values.
        raw = pd.read_csv(source_path, sep=",", dtype=str, encoding="utf-8-sig")
        if not {date_column, value_column}.issubset(raw.columns):
            raw = pd.read_csv(source_path, sep=";", dtype=str, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        raw = pd.read_excel(source_path, sheet_name=sheet_name, dtype=str)
    else:
        raise ValueError("Accepted formats: .csv, .xlsx (old .xls requires conversion)")
    needed = {date_column, value_column}
    if not needed.issubset(raw.columns):
        raise ValueError(f"Required columns {sorted(needed)} absent; available {list(raw.columns)}")
    x = raw[[date_column, value_column]].copy()
    x.columns = ["date", "value"]
    x["date"] = pd.to_datetime(x.date, errors="coerce", dayfirst=False)
    if x.date.isna().any():
        raise ValueError("Unparseable date: use ISO YYYY-MM-DD or YYYY-MM in date column")
    number = x.value.astype("string").str.strip()
    if decimal_comma:
        number = number.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    number = number.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    x["value"] = pd.to_numeric(number, errors="coerce")
    invalid = number.notna() & x.value.isna()
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} unparseable numeric values")
    if not np.isfinite(x.value.dropna().to_numpy(dtype=float)).all():
        raise ValueError("Infinite observations are prohibited")
    normalized_from_nonfirst = int((x.date.dt.day != 1).sum())
    x["month"] = x.date.dt.to_period("M").dt.to_timestamp()
    duplicates = int(len(x) - x.month.nunique())
    if duplicates:
        if aggregation is None:
            raise ValueError(f"Found {duplicates} extra observations within calendar months; specify aggregation explicitly (mean/sum/last)")
        x = x.sort_values("date")
        if aggregation == "last":
            # Prefer last nonmissing measured observation where available.
            clean = x.dropna(subset=["value"]).groupby("month").tail(1)
            monthly = clean.groupby("month")["value"].first()
        elif aggregation == "mean":
            monthly = x.groupby("month")["value"].mean()
        else:
            # min_count prevents treating entirely missing months as zero.
            monthly = x.groupby("month")["value"].sum(min_count=1)
        x = pd.DataFrame({"month": monthly.index, "value": monthly.to_numpy()})
    x = x.sort_values("month")
    y = pd.DataFrame({
        "series_id": series_id,
        "date": x.month.dt.strftime("%Y-%m-%d"),
        "value": x.value.to_numpy(dtype=float),
        "unit": unit,
        "asset_type": asset_type,
        "data_status": data_status,
        "source": source_description,
    })
    if capacity is not None and asset_type != "reservoir":
        raise ValueError("capacity is supported only for reservoir storage")
    if capacity is not None:
        y["capacity"] = float(capacity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y.to_csv(output_path, index=False, float_format="%.15g")
    parsed = load_series(output_path)
    return {
        "input_filename": source_path.name,
        "normalized_filename": output_path.name,
        "asset_type": parsed.asset_type,
        "unit": parsed.unit,
        "source": parsed.source,
        "data_status": parsed.data_status,
        "n_source_rows": int(len(raw)),
        "n_calendar_months": int(len(parsed.values)),
        "n_valid_observations": parsed.n_valid,
        "n_aggregated_extra_rows": duplicates,
        "n_dates_normalized_to_first_day_of_month": normalized_from_nonfirst,
        "aggregation": aggregation,
        "decimal_comma": decimal_comma,
        "capacity": capacity,
    }
