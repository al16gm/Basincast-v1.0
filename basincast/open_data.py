"""Optional open-data historical meteorology: no dependencies for forecasting.

Queries Open-Meteo archive (reanalysis) by point coordinates, and aggregates
complete daily records to monthly precipitation totals and temperature means.
Spatially averaging a whole river basin is outside this point-based helper.
"""
from __future__ import annotations

from datetime import date
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'


def get_open_meteo_monthly(latitude: float, longitude: float, start_date: str,
                           end_date: str, *, model: str = 'era5', timeout: int = 30) -> tuple[pd.DataFrame, dict]:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError('Invalid WGS84 coordinates')
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError('start_date after end_date')
    if model not in ('era5','era5_land'):
        raise ValueError('Choose era5 or era5_land explicitly')
    query={
        'latitude':latitude, 'longitude':longitude,
        'start_date':str(start),'end_date':str(end),
        'daily':'precipitation_sum,temperature_2m_mean',
        'models':model,'timezone':'UTC',
    }
    url=ARCHIVE_URL+'?'+urlencode(query)
    with urlopen(url,timeout=timeout) as response:
        data=json.loads(response.read().decode('utf-8'))
    if data.get('error') or 'daily' not in data:
        raise ValueError(f'Open-Meteo returned an error: {data.get("reason", "no daily data")}')
    daily=data['daily']
    required={'time','precipitation_sum','temperature_2m_mean'}
    if not required.issubset(daily):
        raise ValueError('Open-Meteo daily response is incomplete')
    n=len(daily['time'])
    if n==0 or any(len(daily[key]) != n for key in required):
        raise ValueError('Inconsistent meteorological arrays')
    frame=pd.DataFrame({'date':pd.to_datetime(daily['time']),
                        'precip_mm':pd.to_numeric(daily['precipitation_sum'],errors='coerce'),
                        'tmean_c':pd.to_numeric(daily['temperature_2m_mean'],errors='coerce')})
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError('Invalid or repeated meteorological dates')
    if (frame.precip_mm.dropna() < 0).any():
        raise ValueError('Negative daily precipitation')
    frame['month']=frame.date.dt.to_period('M').dt.to_timestamp()
    monthly=frame.groupby('month').agg(
        precip_mm=('precip_mm',lambda s: s.sum(min_count=1)),
        tmean_c=('tmean_c','mean'),
        n_daily=('date','size'),
        n_valid_precip=('precip_mm','count'),
        n_valid_temp=('tmean_c','count')).reset_index().rename(columns={'month':'date'})
    expected=monthly.date.dt.days_in_month.to_numpy()
    complete=(monthly.n_daily.to_numpy()==expected)&(monthly.n_valid_precip.to_numpy()==expected)&(monthly.n_valid_temp.to_numpy()==expected)
    # Never treat partial months or missing days as fully observed monthly weather.
    monthly.loc[~complete,['precip_mm','tmean_c']]=np.nan
    result=monthly[['date','precip_mm','tmean_c']].copy()
    result['date']=result.date.dt.strftime('%Y-%m-%d')
    return result, {
        'provider':'Open-Meteo historical archive','dataset_model':model,
        'data_kind':'REANALYSIS_NOT_STATION_OBSERVATION',
        'latitude':latitude,'longitude':longitude,
        'start_date':start_date,'end_date':end_date,'request_url':url,
        'n_complete_calendar_months':int(complete.sum()),
        'n_incomplete_or_missing_months':int((~complete).sum()),
        'source_url':'https://open-meteo.com/en/docs/historical-weather-api',
        'license_note':'Check Open-Meteo terms and commercial API requirements before deployment',
        'spatial_note':'Grid point is not necessarily representative of an entire catchment',
    }
