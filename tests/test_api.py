"""End-to-end API checks against the frozen BasinCast prediction engine."""
from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pandas as pd
from fastapi.testclient import TestClient
import pytest

from api import main as service

CLIENT = TestClient(service.app)
EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'river_2_columns.csv'


def test_health_and_info():
    result = CLIENT.get('/health')
    assert result.status_code == 200
    assert result.json()['engine_version'] == '1.0.0'
    info = CLIENT.get('/v1/info').json()
    assert info['max_horizon_months'] == 48
    assert info['external_meteorology_required'] is False


def test_forecast_real_river_json():
    with EXAMPLE.open('rb') as handle:
        response = CLIENT.post(
            '/v1/forecast',
            files={'file': ('river.csv', handle, 'text/csv')},
            data={'series_id': 'River_8107', 'asset_type': 'river', 'unit': 'hm3/month',
                  'max_horizon_months': '3'},
        )
    assert response.status_code == 200, response.text[:900]
    result = response.json()
    assert result['summary']['series_id'] == 'River_8107'
    assert len(result['metrics']) == 3
    assert len(result['forecast']) == 3
    assert len(result['recommended_forecast']) == result['summary']['n_recommended_months']
    assert all(row['is_recommended'] for row in result['recommended_forecast'])
    assert all(row['reliability_label'] != 'BASELINE_SUPPORTED' for row in result['metrics'])
    assert 'result_zip' not in result['summary']
    assert len(result['zip_sha256']) == 64


def test_export_zip_and_integrity():
    # 36 months triggers the explicit short-series baseline protocol.
    df = pd.read_csv(EXAMPLE).iloc[:36]
    payload = df.to_csv(index=False).encode('utf-8')
    result = CLIENT.post(
        '/v1/forecast.zip',
        files={'file': ('short.csv', io.BytesIO(payload), 'text/csv')},
        data={'series_id': 'Short_01', 'asset_type': 'river', 'unit': 'm3/s',
              'max_horizon_months': '6'},
    )
    assert result.status_code == 200, result.text[:800] if result.status_code != 200 else ''
    assert result.headers['content-type'] == 'application/zip'
    with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
        names = set(archive.namelist())
        assert {'summary.json', 'manifest.json', 'metrics_by_horizon.csv',
                'forecast_all_candidates.csv', 'recommended_forecast.csv',
                'normalized_input.csv', 'SHA256SUMS.txt'}.issubset(names)
        summary = json.loads(archive.read('summary.json'))
        assert summary['n_calendar_months'] == 36
        assert summary['n_recommended_months'] == 0


def test_invalid_extensions_and_empty_file():
    wrong = CLIENT.post('/v1/forecast',files={'file':('not_csv.txt', b'hi','text/plain')})
    assert wrong.status_code == 415
    empty = CLIENT.post('/v1/forecast',files={'file':('empty.csv', b'','text/csv')})
    assert empty.status_code == 422


def test_no_access_when_key_is_wrong(monkeypatch):
    monkeypatch.setattr(service,'API_KEY','test_secret')
    monkeypatch.setattr(service,'REQUIRE_API_KEY',True)
    with EXAMPLE.open('rb') as handle:
        result = CLIENT.post('/v1/forecast', files={'file':('river.csv',handle,'text/csv')})
    assert result.status_code == 401
    with EXAMPLE.open('rb') as handle:
        result = CLIENT.post('/v1/forecast',headers={'X-API-Key':'test_secret'},
            files={'file':('river.csv',handle,'text/csv')}, data={'max_horizon_months':'1'})
    assert result.status_code == 200


def test_finite_upload_limit(monkeypatch):
    monkeypatch.setattr(service,'MAX_UPLOAD_BYTES',100)
    result = CLIENT.post('/v1/forecast',files={'file':('river.csv', b'x'*101,'text/csv')})
    assert result.status_code == 413


def test_bad_horizon_and_invalid_series_id():
    response = CLIENT.post('/v1/forecast',files={'file':('river.csv',b'date,value\n2019-01-01,2\n','text/csv')},
                           data={'max_horizon_months':'49'})
    assert response.status_code == 422
    response = CLIENT.post('/v1/forecast',files={'file':('river.csv',b'date,value\n2019-01-01,2\n','text/csv')},
                           data={'series_id':'../../escape'})
    assert response.status_code == 422


def test_csv_wrong_schema_returns_422():
    response = CLIENT.post('/v1/forecast',files={'file':('river.csv',b'foo,bar\na,b\n','text/csv')})
    assert response.status_code == 422
