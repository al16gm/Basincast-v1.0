"""Acceptance tests for automatic, continuous KGE-based reliability."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from basincast.engine import adaptive_config, run_series, run_short_series
from basincast.schema import SeriesData, load_series
from basincast.input import normalize_user_file
from basincast.metrics import rmse

ROOT = Path(__file__).resolve().parents[1]


def test_chronological_protocol_does_not_change_with_analysis_length():
    configs = [adaptive_config(432, max_horizon=h) for h in (3, 12, 24, 48)]
    assert len({(c.test_months, c.validation_months, c.min_prevalidation_months)
                for c in configs}) == 1
    case = load_series(ROOT / 'examples' / 'river_jucar_real.csv')
    short = run_series(case, configs[0])
    long = run_series(case, configs[-1])
    for name in ('kge', 'kge_validation', 'rmse', 'nse', 'quality_supported',
                 'support_reason', 'selected_model', 'validation_selected_baseline'):
        pd.testing.assert_series_equal(short['metrics'][name], long['metrics'][name].iloc[:3],
                                       check_names=True, check_exact=True)
    for name in ('predicted_value', 'forecast_phase', 'selected_model'):
        pd.testing.assert_series_equal(short['forecast'][name], long['forecast'][name].iloc[:3],
                                       check_names=True, check_exact=True)
    assert short['audit']['outer_test_start'] == long['audit']['outer_test_start']
    assert short['audit']['calibration_validation_start'] == long['audit']['calibration_validation_start']


def test_first_kge_failure_is_final_even_if_kge_recovers():
    dates = pd.date_range('1988-01-01', periods=432, freq='MS')
    vals = 10 + 2 * np.sin(np.arange(432) * 2 * np.pi / 12)
    case = SeriesData('Rebound', 'river', 'm3/s', 'SYNTHETIC_DEMO', 'test',
                      pd.Series(vals, index=dates), 'fixture.csv')
    calls = 0

    def controlled_kge(_truth, _forecast):
        nonlocal calls
        calls += 1
        horizon = (calls - 1) // 3 + 1
        # Three KGE calculations per lead: chosen test, baseline test, validation.
        if horizon == 3 and (calls - 1) % 3 == 0:
            return 0.55
        return 0.90

    with patch('basincast.engine.kge', side_effect=controlled_kge):
        result = run_series(case, adaptive_config(len(vals), max_horizon=5))
    assert result['metrics'].quality_supported.tolist() == [True, True, False, True, True]
    assert result['forecast'].forecast_phase.tolist() == [
        'PREDICTION', 'PREDICTION', 'PLANNING_GUIDANCE',
        'PLANNING_GUIDANCE', 'PLANNING_GUIDANCE',
    ]
    assert result['audit']['empirical_contiguous_horizon_months'] == 2
    assert result['audit']['first_unsupported_horizon'] == {
        'month': 3, 'reason': 'TEST_KGE_BELOW_THRESHOLD',
    }
    assert result['forecast'].is_supported_prediction.tolist() == [True, True, False, False, False]


def test_missing_support_is_not_misreported_as_poor_kge():
    dates = pd.date_range('2000-01-01', periods=24, freq='MS')
    vals = pd.Series(np.linspace(10, 15, 24), index=dates)
    case = SeriesData('Short', 'aquifer', 'm', 'USER_PROVIDED_UNVERIFIED',
                      'test', vals, 'fixture.csv')
    result = run_short_series(case, max_horizon=48)
    assert result['audit']['first_unsupported_horizon']['reason'] == 'INSUFFICIENT_RECORD_LENGTH'
    assert result['metrics'].kge.isna().all()
    assert result['metrics'].support_reason.eq('INSUFFICIENT_RECORD_LENGTH').all()
    assert result['forecast'].forecast_phase.eq('PLANNING_GUIDANCE').all()


def test_per_horizon_baselines_are_reported_independently():
    case = load_series(ROOT / 'examples' / 'river_jucar_real.csv')
    result = run_series(case, adaptive_config(len(case.values), max_horizon=3))
    pairs = result['backtest_pairs']
    metrics = result['metrics']
    for h in (1, 2, 3):
        items = pairs[pairs.horizon_months == h]
        row = metrics.loc[metrics.horizon_months == h].iloc[0]
        assert row.rmse_persistence == pytest.approx(rmse(items.y_true, items.y_persistence))
        assert row.rmse_seasonal == pytest.approx(rmse(items.y_true, items.y_seasonal))
        assert row.rmse_skill_vs_persistence == pytest.approx(1-row.rmse / row.rmse_persistence)
        assert row.rmse_skill_vs_seasonal == pytest.approx(1-row.rmse / row.rmse_seasonal)


def test_csv_metadata_rejects_spreadsheet_formula_injection():
    with tempfile.TemporaryDirectory() as work:
        raw = Path(work) / 'data.csv'
        raw.write_text('date,value\n2022-01-01,2\n2022-02-01,3\n', encoding='utf-8')
        for kwargs in ({'unit': '=HYPERLINK("https://example.org")'},
                       {'source_description': '+SUM(1,2)'},
                       {'source_description': '@SUM(1,2)'}):
            settings = {'series_id': 'Safe', 'asset_type': 'river', 'unit': 'm3/s'}
            settings.update(kwargs)
            with pytest.raises(ValueError, match='Unsafe'):
                normalize_user_file(raw, Path(work) / 'normalized.csv', **settings)


def test_automatic_api_response_has_no_manual_horizon_field():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    openapi = client.get('/openapi.json').json()
    body = openapi['paths']['/v1/forecast']['post']['requestBody']
    schema = body['content']['multipart/form-data']['schema']
    properties = openapi['components']['schemas'][schema['$ref'].split('/')[-1]]['properties']
    assert 'max_horizon_months' not in properties
    assert client.get('/v1/info').json()['analysis_ceiling_months'] == 48
