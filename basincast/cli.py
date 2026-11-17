"""BasinCast Engine v1.0 CLI; pure forecast core plus optional climate deltas."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .climate import apply_climate_delta_csv
from .engine import adaptive_config, run_series, run_short_series
from .input import normalize_user_file
from .open_data import get_open_meteo_monthly
from .schema import load_series


def file_sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_forecast(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    series_id: str,
    asset_type: str,
    unit: str,
    max_horizon: int = 12,
    kge_threshold: float = 0.60,
    date_column: str = 'date',
    value_column: str = 'value',
    source_description: str = 'USER_UPLOAD_UNVERIFIED',
    data_status: str = 'USER_PROVIDED_UNVERIFIED',
    aggregation: str | None = None,
    decimal_comma: bool = False,
    capacity: float | None = None,
    fixed_test_months: int | None = None,
    fixed_validation_months: int | None = None,
) -> dict:
    """Prepare, evaluate, and export one case. Returns JSON-serializable summary.

    The public app should show only recommended_forecast.csv as a supported
    trajectory. forecast_all_candidates.csv contains diagnostic extrapolations.
    """
    source_path=Path(input_path)
    out=Path(output_dir)
    if out.exists():
        if out.is_file() or any(out.iterdir()):
            raise ValueError(f'Output exists and is not empty: {out}. Use a new directory to preserve previous runs')
    out.mkdir(parents=True,exist_ok=True)
    normalized=out/'normalized_input.csv'
    io_audit=normalize_user_file(
        source_path,normalized,series_id=series_id,asset_type=asset_type,
        unit=unit,date_column=date_column,value_column=value_column,
        source_description=source_description,data_status=data_status,
        aggregation=aggregation,decimal_comma=decimal_comma,capacity=capacity)
    case=load_series(normalized)
    short_mode = len(case.values) < 60
    if len(case.values) < 12:
        raise ValueError("Need at least 12 calendar months even for unvalidated seasonal baseline")
    if short_mode:
        cfg = None
    else:
        cfg=adaptive_config(len(case.values),max_horizon=max_horizon,kge_threshold=kge_threshold)
    if (fixed_test_months is None) != (fixed_validation_months is None):
        raise ValueError('Specify both fixed test and fixed validation months, or neither')
    if fixed_test_months is not None:
        if short_mode:
            raise ValueError("Cannot run chronological validation with fewer than 60 months")
        from dataclasses import replace
        cfg=replace(cfg,test_months=fixed_test_months,validation_months=fixed_validation_months)
    result=run_short_series(case,max_horizon=max_horizon) if short_mode else run_series(case,cfg)
    result['backtest_pairs'].to_csv(out/'backtest_pairs.csv',index=False)
    result['metrics'].to_csv(out/'metrics_by_horizon.csv',index=False)
    result['forecast'].to_csv(out/'forecast_all_candidates.csv',index=False)
    recommended=result['forecast'][result['forecast'].is_recommended].copy()
    recommended.to_csv(out/'recommended_forecast.csv',index=False)
    (out/'evaluation_audit.json').write_text(json.dumps(result['audit'],ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    summary={
        'software':'BasinCast Engine','version':__version__,
        'series_id':case.series_id,'asset_type':case.asset_type,'unit':case.unit,
        'data_status':case.data_status,'source_description':case.source,
        'n_calendar_months':int(len(case.values)),'n_valid_observations':case.n_valid,
        'issue_date':result['audit']['last_known_date'],
        'max_horizon_requested_months':max_horizon,
        'max_contiguous_supported_horizon_months':result['audit']['empirical_contiguous_horizon_months'],
        'isolated_supported_horizons_months':result['metrics'].loc[result['metrics'].quality_supported,'horizon_months'].astype(int).tolist(),
        'n_recommended_months':int(len(recommended)),
        'n_candidate_months':int(len(result['forecast'])),
        'n_backtest_pairs':int(len(result['backtest_pairs'])),
        'data_note':'A point forecast does not imply calibrated uncertainty or guaranteed reliability',
        'model_skill_note':'KGE quality is distinct from Ridge RMSE improvement relative to baseline',
        'weather_note':'Hydrological forecast uses ONLY historical target values; no external weather, demand or IPCC input is required',
        'scenario_note':'Climate projection adapter is separate and does not imply hydrological impact',
        'io':io_audit,'configuration':asdict(cfg) if cfg is not None else {'mode':'SHORT_UNVALIDATED_SEASONAL_BASELINE','max_horizon':max_horizon},
    }
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    code_root=Path(__file__).resolve().parent
    included=['__init__.py','__main__.py','engine.py','schema.py','metrics.py','input.py','open_data.py','climate.py','cli.py']
    input_reference={'input_filename':source_path.name,'input_sha256':file_sha256(source_path),
                     'normalized_sha256':file_sha256(normalized)}
    manifest={
        'software':'BasinCast Engine','version':__version__,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'raw_input':input_reference,
        'code_sha256':{name:file_sha256(code_root/name) for name in included},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,
                       'pandas':pd.__version__,'scikit_learn':sklearn.__version__},
        'files_sha256':{},
    }
    for path in sorted(out.iterdir()):
        if path.is_file():
            manifest['files_sha256'][path.name]=file_sha256(path)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(manifest['files_sha256'].items())),encoding='ascii')
    zip_path=out.parent/f'{out.name}.zip'
    if zip_path.exists():
        raise ValueError(f'ZIP already exists: {zip_path}. Choose a fresh output name')
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file():archive.write(path,path.name)
    summary['result_zip']=str(zip_path.resolve())
    return summary


def main(argv: list[str] | None = None) -> None:
    p=argparse.ArgumentParser(description='BasinCast Engine v1.0: monthly time series forecasting without required exogenous data')
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('forecast',help='Forecast any monthly resource or demand using one CSV or Excel')
    f.add_argument('--input',required=True)
    f.add_argument('--out',default='results/run1')
    f.add_argument('--series-id',required=True)
    f.add_argument('--type',dest='asset_type',choices=['river','aquifer','reservoir','agricultural_demand','urban_demand','custom'],required=True)
    f.add_argument('--unit',required=True)
    f.add_argument('--max-horizon',type=int,default=12)
    f.add_argument('--kge-threshold',type=float,default=0.6)
    f.add_argument('--date-column',default='date')
    f.add_argument('--value-column',default='value')
    f.add_argument('--source',default='USER_UPLOAD_UNVERIFIED')
    f.add_argument('--data-status',default='USER_PROVIDED_UNVERIFIED',choices=['OBSERVED_REPORTED','USER_PROVIDED_UNVERIFIED','ESTIMATED','RECONSTRUCTED','SYNTHETIC_DEMO'])
    f.add_argument('--aggregation',choices=['mean','sum','last'],default=None)
    f.add_argument('--decimal-comma',action='store_true')
    f.add_argument('--capacity',type=float,default=None,help='Optional maximum reservoir storage in the same volume unit')
    f.add_argument('--test-months',type=int,default=None)
    f.add_argument('--validation-months',type=int,default=None)
    c=sub.add_parser('climate-delta',help='Apply published IPCC/CMIP6 changes to open historical meteorology (not hydrological forecasts)')
    c.add_argument('--historical-meteo',required=True)
    c.add_argument('--published-deltas',required=True)
    c.add_argument('--baseline-start',type=int,required=True)
    c.add_argument('--baseline-end',type=int,required=True)
    c.add_argument('--from-month',required=True)
    c.add_argument('--to-month',required=True)
    c.add_argument('--out',default='results/climate_scenarios.csv')
    o=sub.add_parser('fetch-meteo',help='Fetch historical point meteorology from open reanalysis; optional')
    o.add_argument('--lat',type=float,required=True)
    o.add_argument('--lon',type=float,required=True)
    o.add_argument('--start',required=True)
    o.add_argument('--end',required=True)
    o.add_argument('--model',default='era5',choices=['era5','era5_land'])
    o.add_argument('--out',default='results/historical_meteo.csv')
    args=p.parse_args(argv)
    if args.command=='forecast':
        report=run_forecast(args.input,args.out,series_id=args.series_id,asset_type=args.asset_type,
                            unit=args.unit,max_horizon=args.max_horizon,kge_threshold=args.kge_threshold,
                            date_column=args.date_column,value_column=args.value_column,
                            source_description=args.source,data_status=args.data_status,
                            aggregation=args.aggregation,decimal_comma=args.decimal_comma,capacity=args.capacity,
                            fixed_test_months=args.test_months,fixed_validation_months=args.validation_months)
        print(json.dumps(report,indent=2,ensure_ascii=False))
    elif args.command=='climate-delta':
        result=apply_climate_delta_csv(args.historical_meteo,args.published_deltas,
            baseline_start_year=args.baseline_start,baseline_end_year=args.baseline_end,
            output_start=args.from_month,output_end=args.to_month)
        output=Path(args.out)
        output.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(output,index=False)
        climate_manifest={
            'method':'PUBLISHED_DETERMINISTIC_DELTA_CHANGE',
            'historical_input':str(args.historical_meteo),
            'historical_sha256':file_sha256(args.historical_meteo),
            'published_deltas_input':str(args.published_deltas),
            'published_deltas_sha256':file_sha256(args.published_deltas),
            'output_sha256':file_sha256(output),
            'baseline_reference_period':f'{args.baseline_start}-{args.baseline_end}',
            'output_period':[args.from_month,args.to_month],
            'interpretation':'NOT A MONTHLY WEATHER FORECAST OR A HYDROLOGICAL FORECAST',
        }
        output.with_suffix('.manifest.json').write_text(json.dumps(climate_manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        print('CLIMATE_SCENARIOS_CSV',output.resolve(),'N_ROWS',len(result))
    else:
        meteo,provenance=get_open_meteo_monthly(args.lat,args.lon,args.start,args.end,model=args.model)
        output=Path(args.out)
        output.parent.mkdir(parents=True,exist_ok=True)
        meteo.to_csv(output,index=False)
        provenance['export_sha256']=file_sha256(output)
        output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
        print('HISTORICAL_METEO_CSV',output.resolve(),'N_MONTHS',len(meteo))

if __name__=='__main__':
    main()
