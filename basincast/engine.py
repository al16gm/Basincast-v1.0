"""Direct-horizon monthly forecasts with automatic reliability classification.

Forecasts are direct rather than recursive. Each lead has an independent
chronological evaluation; the evaluation split is independent of the number
of horizons returned. The continuous supported interval ends at the first
horizon with insufficient evidence or KGE below the specified threshold.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .schema import SeriesData, NONNEGATIVE
from .metrics import rmse, kge, nse, disjoint_origin_windows

ANALYSIS_CEILING_MONTHS = 48
EVALUATION_PROTOCOL_ID = "BC_DIRECT_AUTO_HORIZON_1_1"


@dataclass(frozen=True)
class EngineConfig:
    max_horizon: int = ANALYSIS_CEILING_MONTHS
    test_months: int = 60
    validation_months: int = 24
    min_prevalidation_months: int = 60
    min_ridge_training_pairs: int = 36
    min_validation_pairs: int = 8
    min_evaluation_pairs: int = 20
    min_disjoint_windows: int = 4
    kge_threshold: float = 0.60
    ridge_alpha: float = 10.0
    min_ridge_validation_gain_fraction: float = 0.01


def _finite_mean(a: np.ndarray) -> float:
    f = a[np.isfinite(a)]
    return float(np.mean(f)) if f.size else float('nan')


def _feature(values: np.ndarray, dates: pd.DatetimeIndex, t: int, h: int) -> list[float]:
    target_month = int(dates[t+h].month)
    history = values[t-11:t+1]
    return [values[t], values[t-1], values[t-11], values[t-12],
            _finite_mean(values[t-2:t+1]), _finite_mean(history),
            np.sin(2*np.pi*target_month/12), np.cos(2*np.pi*target_month/12)]


def _samples(values: np.ndarray, dates: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for t in range(12, len(values)-h):
        if not np.isfinite(values[t]) or not np.isfinite(values[t+h]):
            continue
        seasonal_offset = (12 - h % 12) % 12
        seasonal = values[t-seasonal_offset]
        fallback = not np.isfinite(seasonal)
        records.append({"origin_index": t, "target_index": t+h,
                        "origin_date": dates[t], "target_date": dates[t+h],
                        "y_true": values[t+h], "persistence": values[t],
                        "seasonal": values[t] if fallback else seasonal,
                        "seasonal_fallback": fallback,
                        "features": _feature(values, dates, t, h)})
    return pd.DataFrame(records, columns=["origin_index", "target_index", "origin_date", "target_date",
                                          "y_true", "persistence", "seasonal", "seasonal_fallback", "features"])


def _ridge() -> Any:
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=10.0))


def _predict_ridge(fit_set: pd.DataFrame, eval_set: pd.DataFrame,
                   alpha: float) -> np.ndarray | None:
    if fit_set.empty or eval_set.empty:
        return None
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(), Ridge(alpha=alpha))
    x_train = np.array(fit_set.features.tolist(), dtype=float)
    model.fit(x_train, fit_set.y_true.to_numpy(dtype=float))
    return np.asarray(model.predict(np.array(eval_set.features.tolist(), dtype=float)), dtype=float)


def _bound_predictions(pred: np.ndarray, case: SeriesData) -> tuple[np.ndarray, int]:
    adjusted = pred.copy()
    if case.asset_type in NONNEGATIVE:
        adjusted = np.maximum(adjusted, 0.0)
    if case.asset_type == 'reservoir' and case.capacity is not None:
        adjusted = np.minimum(adjusted, case.capacity)
    return adjusted, int((np.abs(adjusted - pred) > 1e-10).sum())


def _select_candidate(validation: pd.DataFrame, val_ridge: np.ndarray | None,
                      case: SeriesData, cfg: EngineConfig) -> tuple[str, str, dict[str, float]]:
    if validation.empty:
        return 'seasonal', 'seasonal', {}
    true = validation.y_true.to_numpy(dtype=float)
    perf = {"persistence": rmse(true, validation.persistence.to_numpy(dtype=float)),
            "seasonal": rmse(true, validation.seasonal.to_numpy(dtype=float))}
    base = min(perf, key=perf.get)
    if val_ridge is None or len(validation) < cfg.min_validation_pairs:
        return base, base, perf
    adjusted, _ = _bound_predictions(val_ridge, case)
    perf['ridge'] = rmse(true, adjusted)
    chosen = ('ridge' if perf['ridge'] < perf[base]*(1-cfg.min_ridge_validation_gain_fraction)
              else base)
    return chosen, base, perf


def _get_pred(candidate: str, df: pd.DataFrame, ridge: np.ndarray | None) -> np.ndarray:
    if candidate == 'ridge':
        if ridge is None:
            raise ValueError('No trained ridge model available')
        return ridge
    return df[candidate].to_numpy(dtype=float)


def _safe_float(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def run_series(case: SeriesData, config: EngineConfig | None = None):
    """Return dict with holdout predictions, metrics, issued forecast and provenance.

    All feature preprocessing, hyperparameter settings and model choice are
    locked before the holdout. Evaluating the holdout requires target observed.
    """
    cfg = config or EngineConfig()
    if cfg.max_horizon < 1 or cfg.max_horizon > ANALYSIS_CEILING_MONTHS:
        raise ValueError('max_horizon must be in [1,48]')
    if cfg.test_months < 24 or cfg.validation_months < 12:
        raise ValueError('Require >=24 test months and >=12 validation months')
    values = case.values.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(case.values.index)
    n = len(values)
    split = n-cfg.test_months
    val_start = split-cfg.validation_months
    if val_start < cfg.min_prevalidation_months:
        raise ValueError(f'Insufficient record: need at least {cfg.min_prevalidation_months+cfg.test_months+cfg.validation_months} calendar months')
    if not np.isfinite(values[-1]):
        raise ValueError('Latest month lacks an observed/estimated value; choose an earlier issue date explicitly')
    metrics: list[dict[str, Any]] = []
    eval_pairs: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    for h in range(1, cfg.max_horizon+1):
        all_rows = _samples(values, dates, h)
        # For short histories, long horizons can be computed from the latest
        # seasonal baseline, but must remain explicitly unvalidated.
        train_cal = all_rows[all_rows.target_index < val_start]
        valid = all_rows[(all_rows.origin_index >= val_start) & (all_rows.target_index < split)]
        train_outer = all_rows[all_rows.target_index < split]
        test = all_rows[(all_rows.origin_index >= split) & (all_rows.target_index < n)]
        val_ridge = None
        if len(train_cal) >= cfg.min_ridge_training_pairs and len(valid) >= cfg.min_validation_pairs:
            val_ridge = _predict_ridge(train_cal, valid, cfg.ridge_alpha)
        candidate, baseline, perf = _select_candidate(valid, val_ridge, case, cfg)
        test_ridge = None
        if len(train_outer) >= cfg.min_ridge_training_pairs and not test.empty:
            test_ridge = _predict_ridge(train_outer, test, cfg.ridge_alpha)
        if candidate == 'ridge' and test_ridge is None:
            candidate = baseline
        if test.empty:
            chosen_test = np.array([], dtype=float)
        else:
            chosen_test, nclip = _bound_predictions(_get_pred(candidate, test, test_ridge), case)
        if not test.empty:
            truth = test.y_true.to_numpy(dtype=float)
            baseline_test = test[baseline].to_numpy(dtype=float)
            for j, tup in enumerate(test.itertuples(index=False)):
                eval_pairs.append({"series_id":case.series_id, "asset_type":case.asset_type,
                                   "unit":case.unit,"data_status":case.data_status,
                                   "origin_date":str(tup.origin_date.date()),
                                   "target_date":str(tup.target_date.date()),
                                   "horizon_months":h,"y_true":float(tup.y_true),
                                   "y_pred":float(chosen_test[j]),
                                   "y_persistence":float(tup.persistence),
                                   "y_seasonal":float(tup.seasonal),
                                   "seasonal_fallback":bool(tup.seasonal_fallback),
                                   "selected_model":candidate,
                                   "validation_selected_baseline":baseline})
            rmse_baseline = rmse(truth, baseline_test)
            rmse_chosen = rmse(truth, chosen_test)
            rmse_persistence = rmse(truth, test.persistence.to_numpy(dtype=float))
            rmse_seasonal = rmse(truth, test.seasonal.to_numpy(dtype=float))
            skill_persistence = (1.0 - rmse_chosen / rmse_persistence
                                 if rmse_persistence > 1e-12 else float('nan'))
            skill_seasonal = (1.0 - rmse_chosen / rmse_seasonal
                              if rmse_seasonal > 1e-12 else float('nan'))
            n_windows = disjoint_origin_windows(test.origin_date.to_list(), h)
            kge_chosen = kge(truth, chosen_test)
            kge_baseline = kge(truth, baseline_test)
            validation_truth = valid.y_true.to_numpy(dtype=float)
            validation_pred = _get_pred(candidate, valid, val_ridge)
            validation_pred, _ = _bound_predictions(validation_pred, case)
            validation_kge = kge(validation_truth, validation_pred)
            # KGE is necessary but not sufficient for quality support; require
            # chronological validation, test support and disjoint-origin diagnostic.
            if len(valid) < cfg.min_validation_pairs:
                support_reason = 'INSUFFICIENT_VALIDATION_PAIRS'
            elif len(test) < cfg.min_evaluation_pairs:
                support_reason = 'INSUFFICIENT_TEST_PAIRS'
            elif n_windows < cfg.min_disjoint_windows:
                support_reason = 'INSUFFICIENT_DISJOINT_WINDOWS'
            elif not np.isfinite(validation_kge):
                support_reason = 'UNDEFINED_VALIDATION_KGE'
            elif not np.isfinite(kge_chosen):
                support_reason = 'UNDEFINED_TEST_KGE'
            elif validation_kge < cfg.kge_threshold:
                support_reason = 'VALIDATION_KGE_BELOW_THRESHOLD'
            elif kge_chosen < cfg.kge_threshold:
                support_reason = 'TEST_KGE_BELOW_THRESHOLD'
            else:
                support_reason = 'SUPPORTED'
            quality_supported = support_reason == 'SUPPORTED'
            skill = float(1-rmse_chosen/rmse_baseline) if rmse_baseline > 1e-12 else float('nan')
            incremental_gain = bool(candidate == 'ridge' and np.isfinite(skill) and skill > 0.0)
            if not quality_supported:
                label = 'NOT_DEMONSTRATED'
            elif candidate != 'ridge':
                label = 'BASELINE_KGE_SUPPORTED'
            elif incremental_gain:
                label = 'RIDGE_KGE_AND_RMSE_GAIN'
            else:
                label = 'RIDGE_KGE_NO_RMSE_GAIN'
        else:
            rmse_chosen=rmse_baseline=kge_chosen=kge_baseline=skill=validation_kge=float('nan')
            rmse_persistence=rmse_seasonal=skill_persistence=skill_seasonal=float('nan')
            n_windows=0; quality_supported=False; incremental_gain=False
            support_reason='INSUFFICIENT_TEST_PAIRS'
            label='NOT_DEMONSTRATED'; nclip=0
        metrics.append({"series_id":case.series_id, "asset_type":case.asset_type,
                        "data_status":case.data_status,"horizon_months":h,
                        "selected_model":candidate,"validation_selected_baseline":baseline,
                        "validation_rmse_persistence":_safe_float(perf.get('persistence', float('nan'))),
                        "validation_rmse_seasonal":_safe_float(perf.get('seasonal', float('nan'))),
                        "validation_rmse_ridge":_safe_float(perf.get('ridge', float('nan'))),
                        "n_calibration_rows":len(train_cal), "n_validation_pairs":len(valid),
                        "n_training_pairs":len(train_outer),"n_test_pairs":len(test),
                        "n_disjoint_origin_windows_proxy":n_windows,
                        "kge":_safe_float(kge_chosen),
                        "kge_validation":_safe_float(validation_kge),
                        "kge_baseline_on_test":_safe_float(kge_baseline),
                        "quality_supported":bool(quality_supported),
                        "support_reason":support_reason,
                        "ridge_has_rmse_gain":bool(incremental_gain),
                        "rmse":_safe_float(rmse_chosen),
                        "rmse_persistence":_safe_float(rmse_persistence),
                        "rmse_seasonal":_safe_float(rmse_seasonal),
                        "rmse_skill_vs_persistence":_safe_float(skill_persistence),
                        "rmse_skill_vs_seasonal":_safe_float(skill_seasonal),
                        "nse":_safe_float(nse(test.y_true, chosen_test)) if len(test) else None,
                        "rmse_validation_selected_baseline_on_test":_safe_float(rmse_baseline),
                        "rmse_skill_vs_validation_selected_baseline":_safe_float(skill),
                        "n_pred_clipped":nclip,"reliability_label":label})
        # Forecast issued after the final known month. Retrain selected model on
        # ALL available history only after holdout metrics have been locked.
        if candidate == 'ridge':
            if len(all_rows) < cfg.min_ridge_training_pairs:
                raise ValueError(f'Cannot refit ridge for horizon {h}')
            future_date = dates[-1] + pd.DateOffset(months=h)
            dates_future = dates.append(pd.DatetimeIndex([dates[-1] + pd.DateOffset(months=m) for m in range(1, h+1)]))
            latest_feat = np.array([_feature(values, dates_future, n-1, h)], dtype=float)
            fit_x = np.array(all_rows.features.tolist(), dtype=float)
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                  StandardScaler(), Ridge(alpha=cfg.ridge_alpha))
            model.fit(fit_x, all_rows.y_true.to_numpy(dtype=float))
            issued = float(model.predict(latest_feat)[0])
        else:
            offset=(12-h % 12)%12
            issued = float(values[-1] if candidate=='persistence' or not np.isfinite(values[-1-offset])
                           else values[-1-offset])
            future_date = dates[-1] + pd.DateOffset(months=h)
        adjusted, _ = _bound_predictions(np.array([issued]), case)
        forecasts.append({"series_id":case.series_id,"asset_type":case.asset_type,
                          "unit":case.unit,"issue_date":str(dates[-1].date()),
                          "target_date":str(future_date.date()),"horizon_months":h,
                          "predicted_value":float(adjusted[0]),
                          "selected_model":candidate,"reliability_label":label,
                          "quality_supported":bool(quality_supported),
                          "support_reason":support_reason,
                          "ridge_has_rmse_gain":bool(incremental_gain),
                          "data_status":case.data_status,
                          "is_climate_projection":False,
                          "has_calibrated_predictive_interval":False})
        calibrations.append({"horizon":h,"validation_cutoff":str(dates[split-1].date()),
                             "calibration_window_start":str(dates[val_start].date()),
                             "holdout_start":str(dates[split].date()),
                             "candidate_selection_uses_outer_holdout":False,
                             "training_targets_lte_pretest":True})
    metric_frame = pd.DataFrame(metrics)
    empirical_horizon = 0
    first_failure = None
    for row in metrics:
        if row['quality_supported']:
            empirical_horizon = int(row['horizon_months'])
        else:
            first_failure = {
                'month': int(row['horizon_months']),
                'reason': row['support_reason'],
            }
            break
    issued_forecasts = pd.DataFrame(forecasts)
    supported = issued_forecasts.horizon_months.le(empirical_horizon)
    issued_forecasts['is_supported_prediction'] = supported
    issued_forecasts['is_planning_guidance'] = ~supported
    issued_forecasts['forecast_phase'] = np.where(supported, 'PREDICTION', 'PLANNING_GUIDANCE')
    issued_forecasts['max_contiguous_supported_months'] = empirical_horizon
    issued_forecasts['planning_guidance_note'] = np.where(
        supported,
        '',
        'Exploratory numerical continuation; not a validated prediction or an operational management directive.',
    )
    # A later isolated KGE recovery never restores supported status: the
    # prediction phase must be continuous from the one-month-ahead lead.
    metric_frame['within_contiguous_supported_horizon'] = (
        metric_frame.horizon_months.le(empirical_horizon)
    )
    metric_frame['forecast_phase'] = np.where(
        metric_frame['within_contiguous_supported_horizon'], 'PREDICTION', 'PLANNING_GUIDANCE'
    )
    result = {"backtest_pairs":pd.DataFrame(eval_pairs),"metrics":metric_frame,
              "forecast":issued_forecasts,
              "audit": {"series_id":case.series_id,"asset_type":case.asset_type,
                        "unit":case.unit,"source_file":case.source_file,"source":case.source,
                        "data_status":case.data_status,"n_calendar_months":n,
                        "n_nonmissing_months":case.n_valid,
                        "last_known_date":str(dates[-1].date()),
                        "outer_test_start":str(dates[split].date()),
                        "calibration_validation_start":str(dates[val_start].date()),
                        "n_out_of_sample_predictions":len(eval_pairs),
                        "empirical_contiguous_horizon_months":empirical_horizon,
                        "first_unsupported_horizon":first_failure,
                        "search_ceiling_months":cfg.max_horizon,
                        "search_ceiling_reached_with_full_support":first_failure is None,
                        "horizon_is_provisional":True,
                        "evaluation_protocol_id":EVALUATION_PROTOCOL_ID,
                        "split_selection_depends_on_requested_horizon":False,
                        "intervals_are_calibrated":False,
                        "validation_is_on_reported_observations":case.data_status=='OBSERVED_REPORTED',
                        "nonoverlap_count_is_not_effective_sample_size":True,
                        "algorithm":"DIRECT_HORIZON_RIDGE_OR_TEMPORAL_BASELINES",
                        "quality_rule":"Contiguous lead months with validation and test KGE >= threshold plus minimum chronological support; first failure ends supported prediction phase",
                        "aquifer_kge_caveat":"Absolute-level KGE is sensitive to the reference datum; check errors and persistence skill before management decisions" if case.asset_type == 'aquifer' else None,
                        "incremental_skill_rule":"Ridge selected AND holdout RMSE < validation-selected baseline RMSE",
                        "horizon_is_statistical_not_a_guarantee":True,
                        "physical_bound":"NONNEGATIVE" if case.asset_type in NONNEGATIVE else "NONE",
                        "reservoir_capacity_provided":case.capacity is not None,
                        "config":asdict(cfg),
                        "time_separation":calibrations}}
    return result


def adaptive_config(n_months: int, max_horizon: int = ANALYSIS_CEILING_MONTHS, *,
                    kge_threshold: float = 0.60) -> EngineConfig:
    """Select chronological windows from record length, never forecast request.

    The maximum potentially supportable horizon is inferred from available
    calendar months with a fixed ceiling. The chosen validation/test windows
    therefore stay IDENTICAL when a caller evaluates a subset of leads.
    Returning unsupported horizons is allowed, but they are never designated
    as validated predictions.
    """
    if not 1 <= max_horizon <= ANALYSIS_CEILING_MONTHS:
        raise ValueError('max_horizon must be between 1 and 48 months')
    if not np.isfinite(kge_threshold) or not -1.0 <= kge_threshold <= 1.0:
        raise ValueError('kge_threshold must be a finite number in [-1, 1]')
    if n_months < 60:
        raise ValueError('At least 60 calendar months required for this validated engine; shorter records need a descriptive baseline-only workflow')
    min_pre = 60 if n_months >= 120 else 24
    min_valid = 8 if n_months >= 120 else 6
    max_potential = min(ANALYSIS_CEILING_MONTHS,
                        max(1, (n_months - min_pre - min_valid - 2) // 5))
    # IMPORTANT: max_horizon intentionally DOES NOT influence these windows.
    # This prevents changing first-month KGE by changing the output length.
    evaluated_potential = max_potential
    validation = max(12, evaluated_potential + min_valid)
    test = max(24, 4 * evaluated_potential + 1)
    while n_months - validation - test < min_pre and evaluated_potential > 1:
        evaluated_potential -= 1
        validation = max(12, evaluated_potential + min_valid)
        test = max(24, 4 * evaluated_potential + 1)
    if n_months - validation - test < min_pre:
        raise ValueError('Insufficient history for strict chronological validation')
    return EngineConfig(max_horizon=max_horizon, test_months=test,
                        validation_months=validation,
                        min_prevalidation_months=min_pre,
                        min_validation_pairs=min_valid,
                        kge_threshold=kge_threshold)


def run_short_series(case: SeriesData, max_horizon: int = ANALYSIS_CEILING_MONTHS):
    """Explicit, UNVALIDATED baseline forecasts for 12--59 calendar months.

    No KGE, skill, validation or reliable horizon can be estimated using the
    production time-split protocol. This is not a substitute for run_series.
    """
    if not 1 <= max_horizon <= 48:
        raise ValueError('max_horizon must be in [1,48]')
    values = case.values.to_numpy(dtype=float)
    n=len(values)
    if not 12 <= n < 60:
        raise ValueError('Short-series fallback requires 12..59 calendar months')
    if not np.isfinite(values[-1]):
        raise ValueError('Latest month has no observed value')
    dates=pd.DatetimeIndex(case.values.index)
    forecasts=[];metrics=[]
    for h in range(1,max_horizon+1):
        offset=(12-h%12)%12
        seasonal=values[-1-offset]
        if not np.isfinite(seasonal):
            seasonal=values[-1]
        raw=np.array([float(seasonal)])
        adjusted,nclip=_bound_predictions(raw,case)
        forecasts.append({
            'series_id':case.series_id,'asset_type':case.asset_type,'unit':case.unit,
            'issue_date':str(dates[-1].date()),
            'target_date':str((dates[-1]+pd.DateOffset(months=h)).date()),
            'horizon_months':h,'predicted_value':float(adjusted[0]),
            'selected_model':'seasonal_unvalidated',
            'reliability_label':'NOT_DEMONSTRATED',
            'quality_supported':False,'ridge_has_rmse_gain':False,
            'is_supported_prediction':False,'is_planning_guidance':True,
            'forecast_phase':'PLANNING_GUIDANCE',
            'planning_guidance_note':'Insufficient observed history for chronological validation; not a validated prediction or operational directive.',
            'max_contiguous_supported_months':0,
            'data_status':case.data_status,'is_climate_projection':False,
            'has_calibrated_predictive_interval':False,
        })
        metrics.append({
            'series_id':case.series_id,'asset_type':case.asset_type,
            'data_status':case.data_status,'horizon_months':h,
            'selected_model':'seasonal_unvalidated','validation_selected_baseline':'seasonal',
            'n_calibration_rows':0,'n_validation_pairs':0,'n_training_pairs':0,
            'n_test_pairs':0,'n_disjoint_origin_windows_proxy':0,
            'kge':None,'kge_validation':None,'rmse':None,'nse':None,
            'rmse_skill_vs_validation_selected_baseline':None,
            'quality_supported':False,'support_reason':'INSUFFICIENT_RECORD_LENGTH',
            'ridge_has_rmse_gain':False,
            'rmse_persistence':None,'rmse_seasonal':None,
            'rmse_skill_vs_persistence':None,'rmse_skill_vs_seasonal':None,
            'n_pred_clipped':nclip,'reliability_label':'NOT_DEMONSTRATED',
            'within_contiguous_supported_horizon':False,
            'forecast_phase':'PLANNING_GUIDANCE',
        })
    return {
        'backtest_pairs':pd.DataFrame(columns=['series_id','asset_type','origin_date','target_date','horizon_months','y_true','y_pred']),
        'metrics':pd.DataFrame(metrics),
        'forecast':pd.DataFrame(forecasts),
        'audit':{
            'series_id':case.series_id,'asset_type':case.asset_type,'unit':case.unit,
            'source_file':case.source_file,'source':case.source,
            'data_status':case.data_status,'n_calendar_months':n,
            'n_nonmissing_months':case.n_valid,
            'last_known_date':str(dates[-1].date()),
            'n_out_of_sample_predictions':0,
            'empirical_contiguous_horizon_months':0,
            'first_unsupported_horizon':{'month':1,'reason':'INSUFFICIENT_RECORD_LENGTH'},
            'search_ceiling_months':max_horizon,
            'search_ceiling_reached_with_full_support':False,
            'horizon_is_provisional':True,
            'evaluation_protocol_id':EVALUATION_PROTOCOL_ID,
            'split_selection_depends_on_requested_horizon':False,
            'intervals_are_calibrated':False,
            'algorithm':'UNVALIDATED_SEASONAL_BASELINE_NO_TRAINING',
            'validation_is_on_reported_observations':False,
            'no_kge_available_due_to_short_record':True,
            'min_months_for_validated_mode':60,
            'quality_rule':'NOT_EVALUABLE_WITH_THIS_RECORD',
        },
    }
