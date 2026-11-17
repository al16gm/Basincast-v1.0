"""BasinCast v1.0 regression and anti-leakage tests (offline only)."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from basincast import EngineConfig, SeriesData, adaptive_config, load_series, normalize_user_file, run_series
from basincast.cli import run_forecast
from basincast.climate import apply_climate_deltas
from basincast.metrics import kge, rmse
from basincast.open_data import get_open_meteo_monthly

ROOT=Path(__file__).resolve().parents[1]


class InputAndEngineTests(unittest.TestCase):
    def _raw(self, folder, count=432, *, seasonal=True):
        dates=pd.date_range('1988-01-01',periods=count,freq='MS')
        vals=10+2*np.sin(2*np.pi*(np.arange(count)%12)/12) if seasonal else np.arange(count,dtype=float)
        p=Path(folder)/'raw.csv'
        pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'value':vals}).to_csv(p,index=False)
        return p

    def test_basic_monthly_two_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=self._raw(tmp)
            dst=Path(tmp)/'standard.csv'
            info=normalize_user_file(raw,dst,series_id='WorldRiver_1',asset_type='river',unit='m3/s')
            case=load_series(dst)
            self.assertEqual(info['n_valid_observations'],432)
            self.assertEqual(case.data_status,'USER_PROVIDED_UNVERIFIED')

    def test_48_month_success_when_historical_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=self._raw(tmp)
            target=Path(tmp)/'std.csv'
            normalize_user_file(source,target,series_id='LongSeasonal',asset_type='river',unit='hm3/month')
            case=load_series(target)
            cfg=adaptive_config(len(case.values),max_horizon=48)
            result=run_series(case,cfg)
            self.assertEqual(len(result['forecast']),48)
            self.assertEqual(result['audit']['empirical_contiguous_horizon_months'],48)
            self.assertTrue(result['forecast'].is_recommended.all())
            self.assertTrue(result['metrics'].quality_supported.all())
            self.assertEqual(int(result['metrics'].iloc[-1].n_disjoint_origin_windows_proxy),4)

    def test_partial_support_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=self._raw(tmp,count=70)
            norm=Path(tmp)/'std.csv'
            normalize_user_file(source,norm,series_id='ShortRecord',asset_type='urban_demand',unit='m3/month')
            case=load_series(norm)
            result=run_series(case,adaptive_config(70,max_horizon=48))
            self.assertEqual(len(result['forecast']),48)
            self.assertTrue((result['forecast'].loc[result['forecast'].horizon_months>15,'is_recommended']==False).all())
            self.assertTrue((result['forecast'].loc[result['forecast'].horizon_months>25,'quality_supported']==False).all())

    def test_observational_test_alteration_does_not_reselect(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self._raw(tmp)
            q=Path(tmp)/'std.csv'
            normalize_user_file(p,q,series_id='NoLeak',asset_type='custom',unit='units')
            case=load_series(q)
            cfg=adaptive_config(432,max_horizon=48)
            a=run_series(case,cfg)
            vals=case.values.copy()
            vals.iloc[-cfg.test_months:]+=1000
            altered=SeriesData(case.series_id,case.asset_type,case.unit,case.data_status,case.source,vals,case.source_file,case.capacity)
            b=run_series(altered,cfg)
            self.assertEqual(a['metrics'].selected_model.tolist(),b['metrics'].selected_model.tolist())
            self.assertEqual(a['metrics'].validation_selected_baseline.tolist(),b['metrics'].validation_selected_baseline.tolist())
            np.testing.assert_allclose(a['metrics'].kge_validation.to_numpy(dtype=float),b['metrics'].kge_validation.to_numpy(dtype=float),equal_nan=True,rtol=0,atol=0)

    def test_metrics_differentiation(self):
        case=load_series(ROOT/'examples'/'river_jucar_real.csv')
        out=run_series(case,adaptive_config(len(case.values),max_horizon=12))
        m=out['metrics']
        self.assertFalse((m[m.selected_model=='ridge'].reliability_label=='BASELINE_KGE_SUPPORTED').any())
        for row in m.itertuples(index=False):
            if row.reliability_label=='RIDGE_KGE_AND_RMSE_GAIN':
                self.assertTrue(row.quality_supported)
                self.assertGreater(row.rmse_skill_vs_validation_selected_baseline,0)
            if row.reliability_label=='RIDGE_KGE_NO_RMSE_GAIN':
                self.assertTrue(row.quality_supported)
                self.assertLessEqual(row.rmse_skill_vs_validation_selected_baseline,0)

    def test_raw_daily_requires_explicit_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'daily.csv'
            pd.DataFrame({'date':['2020-01-03','2020-01-10','2020-02-01'], 'value':[5,7,2]}).to_csv(raw,index=False)
            with self.assertRaisesRegex(ValueError,'aggregation'):
                normalize_user_file(raw,Path(tmp)/'bad.csv',series_id='A',asset_type='river',unit='m3/s')
            audit=normalize_user_file(raw,Path(tmp)/'good.csv',series_id='A',asset_type='river',unit='m3/s',aggregation='mean')
            self.assertEqual(audit['n_aggregated_extra_rows'],1)
            self.assertEqual(pd.read_csv(Path(tmp)/'good.csv').value.iloc[0],6)

    def test_decimal_comma_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'es.csv'
            p.write_text('date;value\n2020-01-01;"2.911.791,50"\n2020-02-01;"2.004.000,00"\n')
            out=Path(tmp)/'parsed.csv'
            normalize_user_file(p,out,series_id='City',asset_type='urban_demand',unit='m3/month',decimal_comma=True)
            self.assertAlmostEqual(float(pd.read_csv(out).value.iloc[0]),2911791.5)

    def test_reservoir_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=self._raw(tmp,count=120)
            dest=Path(tmp)/'cap.csv'
            normalize_user_file(raw,dest,series_id='R1',asset_type='reservoir',unit='hm3',capacity=20)
            self.assertEqual(load_series(dest).capacity,20)
            with self.assertRaisesRegex(ValueError,'exceeds capacity'):
                normalize_user_file(raw,Path(tmp)/'bad.csv',series_id='R1',asset_type='reservoir',unit='hm3',capacity=10)

    def test_undefined_kge_not_declared_supported(self):
        dates=pd.date_range('1990-01-01',periods=132,freq='MS')
        case=SeriesData('Constant','urban_demand','m3/month','USER_PROVIDED_UNVERIFIED','test',pd.Series(np.ones(132),index=dates),'test')
        result=run_series(case,adaptive_config(132,max_horizon=12))
        self.assertFalse(result['metrics'].quality_supported.any())
        self.assertEqual(result['audit']['empirical_contiguous_horizon_months'],0)

    def test_series_shorter_than_60_rejected(self):
        with self.assertRaisesRegex(ValueError,'At least 60'):
            adaptive_config(59,12)

    def test_horizon_above_48_rejected(self):
        with self.assertRaises(ValueError):adaptive_config(500,49)

    def test_cli_reproducibility_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=self._raw(tmp,count=120)
            report=run_forecast(raw,Path(tmp)/'run',series_id='FieldWell',asset_type='aquifer',unit='m',max_horizon=12)
            self.assertIn('max_contiguous_supported_horizon_months',report)
            with zipfile.ZipFile(report['result_zip']) as bundle:
                manifest=json.loads(bundle.read('manifest.json'))
                for name,digest in manifest['files_sha256'].items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(),digest)
                self.assertIn('recommended_forecast.csv',bundle.namelist())
                self.assertEqual(len(pd.read_csv(bundle.open('forecast_all_candidates.csv'))),12)
            with self.assertRaisesRegex(ValueError,'Output exists'):
                run_forecast(raw,Path(tmp)/'run',series_id='FieldWell',asset_type='aquifer',unit='m')


class ClimateScenarioTests(unittest.TestCase):
    def _climate(self):
        dates=pd.date_range('1995-01-01','2014-12-01',freq='MS')
        h=pd.DataFrame({'date':dates,'precip_mm':np.where(dates.month.isin([6,7,8]),15.0,60.0),
                        'tmean_c':12+10*np.cos(2*np.pi*(dates.month-7)/12)})
        delta=pd.DataFrame([{'scenario':sc,'month':m,'precip_change_pct':deltap,
                             'temperature_change_c':deltat,'reference_period':'1995-2014',
                             'projection_period':'2021-2040','source_url':'https://example.org/TEST_FIXTURE_NOT_IPCC'}
                            for sc,deltap,deltat in [('TEST_LOW',0.,0.),('TEST_HIGH',-20.,2.)]
                            for m in range(1,13)])
        return h,delta

    def test_delta_deterministic_no_stochasticity(self):
        h,d=self._climate()
        a=apply_climate_deltas(h,d,baseline_start_year=1995,baseline_end_year=2014,output_start='2024-01-01',output_end='2025-12-01')
        b=apply_climate_deltas(h,d,baseline_start_year=1995,baseline_end_year=2014,output_start='2024-01-01',output_end='2025-12-01')
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(len(a),48)
        m=a[(a.scenario=='TEST_HIGH') & (a.month==1)].iloc[0]
        self.assertAlmostEqual(m.precip_mm_scenario,48)
        self.assertAlmostEqual(m.tmean_c_scenario,m.tmean_c+2)
        self.assertTrue(a.interpretation.str.contains('NOT_MONTHLY_WEATHER_OR_HYDRO').all())

    def test_missing_ipcc_source_rejected(self):
        h,d=self._climate()
        d.loc[0,'source_url']=''
        with self.assertRaisesRegex(ValueError,'source URL'):
            apply_climate_deltas(h,d,baseline_start_year=1995,baseline_end_year=2014,output_start='2024-01-01',output_end='2025-12-01')

    def test_wrong_reference_period_rejected(self):
        h,d=self._climate()
        with self.assertRaisesRegex(ValueError,'reference_period'):
            apply_climate_deltas(h,d,baseline_start_year=1981,baseline_end_year=2010,output_start='2024-01-01',output_end='2025-12-01')

    def test_monthly_changes_required(self):
        h,d=self._climate()
        with self.assertRaisesRegex(ValueError,'12 monthly'):
            apply_climate_deltas(h,d.iloc[:-1],baseline_start_year=1995,baseline_end_year=2014,output_start='2024-01-01',output_end='2025-12-01')

    def test_open_meteo_adapter_offline_fixture(self):
        days=pd.date_range('2020-01-01','2020-03-20',freq='D')
        mocked={'daily':{'time':days.strftime('%Y-%m-%d').tolist(),
                         'precipitation_sum':[1.0]*len(days),
                         'temperature_2m_mean':[10.0]*len(days)}}
        class HttpResponse:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self):return json.dumps(mocked).encode()
        with patch('basincast.open_data.urlopen',return_value=HttpResponse()):
            result,meta=get_open_meteo_monthly(40,-3,'2020-01-01','2020-03-20')
        self.assertEqual(len(result),3)
        self.assertEqual(float(result.iloc[0].precip_mm),31)
        self.assertEqual(float(result.iloc[1].precip_mm),29)
        self.assertTrue(np.isnan(result.iloc[2].precip_mm))
        self.assertEqual(meta['n_complete_calendar_months'],2)


if __name__=='__main__':unittest.main()

class AdditionalSafetyTests(unittest.TestCase):
    def test_unvalidated_short_baseline_never_supported(self):
        from basincast.engine import run_short_series
        dates=pd.date_range('2020-01-01',periods=24,freq='MS')
        case=SeriesData('Short','river','m3/s','USER_PROVIDED_UNVERIFIED','uploaded',
                        pd.Series(np.arange(24,dtype=float),index=dates),'raw')
        result=run_short_series(case,48)
        self.assertEqual(result['audit']['empirical_contiguous_horizon_months'],0)
        self.assertFalse(result['forecast'].is_recommended.any())
        self.assertEqual(len(result['forecast']),48)
        self.assertTrue(result['metrics'].kge.isna().all())

    def test_missing_target_not_silently_interpolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'gaps.csv'
            pd.DataFrame({'date':['2020-01-01','2020-03-01'],'value':[1,3]}).to_csv(p,index=False)
            dest=Path(tmp)/'std.csv'
            normalize_user_file(p,dest,series_id='Missing',asset_type='aquifer',unit='m')
            case=load_series(dest)
            self.assertEqual(len(case.values),3)
            self.assertTrue(np.isnan(case.values.iloc[1]))

    def test_short_60_months_and_long_candidate_no_crash(self):
        dates=pd.date_range('2010-01-01',periods=60,freq='MS')
        vals=4+np.sin(2*np.pi*np.arange(60)/12)
        case=SeriesData('ShortestEvaluable','aquifer','m','ESTIMATED','test',pd.Series(vals,index=dates),'test.csv')
        result=run_series(case,adaptive_config(60,max_horizon=48))
        self.assertEqual(len(result['forecast']),48)
        self.assertEqual(int(result['metrics'].iloc[-1].n_test_pairs),0)
        self.assertFalse(bool(result['forecast'].iloc[-1].is_recommended))

    def test_synthetic_example_provenance_never_claims_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'r.csv'
            dates=pd.date_range('2000-01-01',periods=75,freq='MS')
            pd.DataFrame({'date':dates,'value':2+np.sin(np.arange(75))}).to_csv(p,index=False)
            report=run_forecast(p,Path(tmp)/'out',series_id='User',asset_type='river',unit='m3/s')
            self.assertEqual(report['data_status'],'USER_PROVIDED_UNVERIFIED')

    def test_no_random_climate_generator_in_public_api(self):
        from basincast import climate
        self.assertFalse(hasattr(climate,'stochastic_generator'))
        self.assertFalse(hasattr(climate,'monte_carlo'))
