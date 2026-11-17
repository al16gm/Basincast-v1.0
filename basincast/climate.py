"""Deterministic IPCC/CMIP6 delta-change scenarios for meteorology only.

This module never treats climate scenarios as a month-by-month meteorological
forecast, and never scales river levels or water demand without a calibrated
climate-to-target model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_DELTAS = {
    "scenario", "month", "precip_change_pct", "temperature_change_c",
    "reference_period", "projection_period", "source_url",
}


def apply_climate_deltas(
    historical_monthly_weather: pd.DataFrame,
    published_changes: pd.DataFrame,
    *,
    baseline_start_year: int,
    baseline_end_year: int,
    output_start: str,
    output_end: str,
    min_years_per_month: int = 5,
) -> pd.DataFrame:
    """Apply published seasonal changes to observed historical monthly normals.

    Historical columns: date, precip_mm (monthly total), tmean_c (monthly mean).
    Published changes: 12 rows per scenario, one for each month. Scenario
    metadata and reference period must be complete. No random sampling.

    IPCC scenario changes describe climatological differences under emissions
    scenarios. The output is a deterministic climate *scenario*, NOT a
    monthly weather forecast and NOT a hydrological simulation.
    """
    needed = {"date", "precip_mm", "tmean_c"}
    if not needed.issubset(historical_monthly_weather.columns):
        raise ValueError(f"Historical climate requires {sorted(needed)}")
    if not REQUIRED_DELTAS.issubset(published_changes.columns):
        raise ValueError(f"Projection requires {sorted(REQUIRED_DELTAS)}")
    if baseline_start_year > baseline_end_year:
        raise ValueError("Baseline years are reversed")
    if baseline_end_year - baseline_start_year + 1 < min_years_per_month:
        raise ValueError("Baseline shorter than minimum historical support")
    d = published_changes.copy()
    if d.empty or d.isna().any().any():
        raise ValueError("Published changes have missing fields")
    d["month"] = pd.to_numeric(d.month, errors="raise").astype(int)
    if not d.month.between(1,12).all():
        raise ValueError("Scenario months must be 1..12")
    if d.duplicated(["scenario", "month"]).any():
        raise ValueError("Duplicate month for a scenario")
    if (d.groupby("scenario").month.nunique() != 12).any():
        raise ValueError("Each scenario requires exactly 12 monthly deltas")
    for col in ("precip_change_pct", "temperature_change_c"):
        d[col] = pd.to_numeric(d[col], errors="raise")
        if not np.isfinite(d[col]).all():
            raise ValueError("Nonfinite climate changes")
    if (d.precip_change_pct < -100).any():
        raise ValueError("Precipitation reduction cannot exceed -100%")
    if not d.source_url.astype(str).str.startswith(("https://", "http://")).all():
        raise ValueError("A valid public source URL is required for all changes")
    expected_ref=f"{baseline_start_year}-{baseline_end_year}"
    if not d.reference_period.astype(str).eq(expected_ref).all():
        raise ValueError(f"IPCC reference_period must match historical baseline {expected_ref}")
    # A given scenario must be a single sourced time slice and source reference.
    for scenario, group in d.groupby('scenario'):
        if group.source_url.nunique() != 1 or group.projection_period.nunique() != 1:
            raise ValueError(f"Scenario {scenario} uses inconsistent source/projection metadata")
    h = historical_monthly_weather.copy()
    h['date'] = pd.to_datetime(h.date, errors='raise')
    if h.date.duplicated().any():
        raise ValueError("Historical weather requires one observation per month")
    h['year'] = h.date.dt.year
    h['month'] = h.date.dt.month
    h = h[h.year.between(baseline_start_year, baseline_end_year)]
    if h.empty:
        raise ValueError("No historical weather within requested IPCC reference period")
    for col in ('precip_mm','tmean_c'):
        h[col] = pd.to_numeric(h[col],errors='raise')
    if (h.precip_mm.dropna()<0).any():
        raise ValueError('Monthly precipitation cannot be negative')
    means = h.groupby('month',as_index=False).agg(
        precip_mm=('precip_mm','mean'), tmean_c=('tmean_c','mean'),
        precip_n=('precip_mm','count'),temp_n=('tmean_c','count'))
    if (len(means) != 12 or (means[['precip_n','temp_n']]<min_years_per_month).any().any()):
        raise ValueError("Insufficient monthly climatology across 12 calendar months")
    starts = pd.date_range(output_start,output_end,freq='MS')
    if starts.empty:
        raise ValueError("Future scenario period is empty")
    target=pd.DataFrame({'target_date':starts,'month':starts.month})
    expanded=target.merge(means,on='month',how='left').merge(d,on='month',how='left')
    # The above left-join naturally creates one row per scenario, month and year.
    expanded['precip_mm_scenario']=(expanded.precip_mm*(1+expanded.precip_change_pct/100)).clip(lower=0)
    expanded['tmean_c_scenario']=expanded.tmean_c+expanded.temperature_change_c
    expanded['method']='PUBLISHED_DELTA_CHANGE_OF_HISTORICAL_MONTHLY_NORMALS'
    expanded['interpretation']='DETERMINISTIC_CLIMATE_SCENARIO_NOT_MONTHLY_WEATHER_OR_HYDRO_FORECAST'
    return expanded[['scenario','target_date','month','precip_mm_scenario','tmean_c_scenario',
                     'precip_mm','tmean_c','precip_change_pct','temperature_change_c',
                     'reference_period','projection_period','source_url','method','interpretation']].sort_values(['scenario','target_date']).reset_index(drop=True)


def apply_climate_delta_csv(historical_csv: str | Path, deltas_csv: str | Path, **kwargs):
    return apply_climate_deltas(pd.read_csv(historical_csv), pd.read_csv(deltas_csv), **kwargs)
