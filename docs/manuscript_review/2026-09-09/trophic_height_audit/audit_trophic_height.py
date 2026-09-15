"""Read-only source audit; writes diagnostics only beside this script."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
ROOT = REPO / 'src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita_neg_const'
RET = ROOT / 'retention_protocol/tukey_iqr_1p5_min50pct_threshold0p50_v1'
METRIC = 'PseudoMeanTrophicHeight'
VALUE = 'PseudoNetworkXMeanTrophicLevel'
STATUS = 'PseudoNetworkXTrophicLevelStatusCode'
FOCUS = ['Dutch Microfauna food web PlotB', 'Ythan Estuary']

def save(frame, name):
    frame.to_csv(OUT / name, index=False, float_format='%.17g')

retention = pd.read_csv(RET / 'retention_by_foodweb_metric.csv')
retention = retention[retention.Metric.eq(METRIC)].copy()
retention['Foodweb'] = retention.Foodweb.str.strip()
assert len(retention) == 290 * 9
assert retention.K.eq(10).all() and retention.Threshold.eq(.5).all()
assert retention.TotalRunUnits.eq(50).all()
assert not retention.duplicated(['Foodweb', 'TrainRatio']).any()
summary = retention.groupby('TrainRatio').agg(
    Foodwebs=('Foodweb', 'size'),
    IncludedFoodwebs=('MeetsMinimumRetainedRuns', 'sum'),
    InvalidRuns=('InvalidRunsBeforeTukey', 'sum'),
    OutlierRuns=('OutlierRunsExcluded', 'sum'),
).reset_index()
summary['ExcludedFoodwebs'] = summary.Foodwebs - summary.IncludedFoodwebs
save(summary, 'retention_all_ratios.csv')
columns = ['Foodweb', 'TrainRatio', 'TotalRunUnits', 'ValidRunsBeforeTukey',
           'InvalidRunsBeforeTukey', 'OutlierRunsExcluded', 'RetainedRunsAfterTukey',
           'MinimumRetainedRuns', 'MeetsMinimumRetainedRuns', 'Q1', 'Q3', 'IQR',
           'LowerFence', 'UpperFence']
save(retention[retention.MeetsMinimumRetainedRuns.eq(0)][columns], 'excluded_foodwebs.csv')
save(retention[retention.Foodweb.isin(FOCUS)][columns], 'focus_foodwebs_all_ratios.csv')

files = sorted((ROOT / 'prediction_scores_logs').glob('*.csv'))
assert len(files) == 290
use = ['Version', 'K', 'TrainRatio', 'Threshold', 'ExperimentID', 'Seed', VALUE, STATUS,
       'EmpiricalNetworkXMeanTrophicLevel', 'EmpiricalNetworkXTrophicLevelStatusCode',
       'PseudoNetworkXTrophicLevelNumSpeciesFull', 'PseudoNetworkXTrophicLevelNumSpeciesLargest']

def read_raw(path):
    frame = pd.read_csv(path, usecols=use)
    frame = frame[np.isclose(frame.Threshold, .5) & frame.K.eq(10)].copy()
    frame['Foodweb'] = path.name.split('_tax_mass_results_')[0].strip()
    frame['SourceFile'] = str(path)
    return frame

with ThreadPoolExecutor(max_workers=8) as pool:
    raw = pd.concat(list(pool.map(read_raw, files)), ignore_index=True)
assert len(raw) == 290 * 9 * 50
assert raw.Version.eq('WLNM_dir_neg').all()
assert not raw.duplicated(['Foodweb', 'TrainRatio', 'ExperimentID']).any()
raw['FiniteValue'] = np.isfinite(raw[VALUE])
raw['FiniteReference'] = np.isfinite(raw.EmpiricalNetworkXMeanTrophicLevel)
assert raw.FiniteReference.all()
assert (raw.FiniteValue == raw[STATUS].eq(0)).all()
counts = raw.groupby(['TrainRatio', STATUS]).size().rename('Runs').reset_index()
save(counts, 'raw_status_all_ratios.csv')
focus = raw[raw.Foodweb.isin(FOCUS)]
save(focus.groupby(['Foodweb', 'TrainRatio', STATUS]).size().rename('Runs').reset_index(), 'focus_status_all_ratios.csv')
save(focus[focus.TrainRatio.eq(60)], 'focus_train60_runs.csv')

# Verify baseline retention independently from every original metric vector.
sensitivity = []
for (web, ratio), group in raw.groupby(['Foodweb', 'TrainRatio']):
    values = group.loc[group.FiniteValue, VALUE].to_numpy()
    before = len(values)
    q1, q3 = np.quantile(values, [.25, .75]) if before else (np.nan, np.nan)
    iqr = q3 - q1
    selected = retention[retention.Foodweb.eq(web) & retention.TrainRatio.eq(ratio)].iloc[0]
    for factor in [1.5, 2., 3., 5., 15., np.inf]:
        kept = before if np.isinf(factor) else int(((values >= q1-factor*iqr) & (values <= q3+factor*iqr)).sum())
        if factor == 1.5:
            assert before == selected.ValidRunsBeforeTukey, (web, ratio, 'finite count')
            assert kept == selected.RetainedRunsAfterTukey, (web, ratio, 'retained count')
        for fraction in [.5, .4, .38]:
            minimum = math.ceil(50 * fraction)
            sensitivity.append(dict(Foodweb=web, TrainRatio=ratio,
                IQRMultiplier='none' if np.isinf(factor) else factor,
                MinimumRetainedFraction=fraction, MinimumRetainedRuns=minimum,
                ValidRunsBeforeTukey=before, RetainedRuns=kept, Included=kept >= minimum))
sensitivity = pd.DataFrame(sensitivity)
save(sensitivity.groupby(['TrainRatio', 'IQRMultiplier', 'MinimumRetainedFraction', 'MinimumRetainedRuns'], dropna=False).Included.sum().rename('IncludedFoodwebs').reset_index(), 'sensitivity_all_ratios.csv')
save(sensitivity[sensitivity.Foodweb.isin(FOCUS) & sensitivity.TrainRatio.eq(60)], 'sensitivity_focus_train60.csv')

# Demonstrate that the hard upper bound is a heuristic, not an existence condition.
A = np.zeros((6, 6))
A[1:, 1:] = 1
np.fill_diagonal(A, 0)
A[0, 1] = 1
M = np.eye(6) - A.T / np.maximum(A.sum(axis=0), 1)[:, None]
levels = np.linalg.solve(M, np.ones(6))
example = dict(Adjacency=A.tolist(), TrophicLevels=levels.tolist(),
    ReciprocalCondition1Norm=1/np.linalg.cond(M, 1),
    MaximumAbsoluteResidual=float(abs(M@levels-np.ones(6)).max()),
    CurrentUpperLimit=20,
    Note='Synthetic demonstration only. Does not establish why historical status-3 rows failed.')
(OUT / 'upper_bound_counterexample.json').write_text(json.dumps(example, indent=2))

source_files = files + [RET/'retention_by_foodweb_metric.csv', RET/'retention_manifest.json',
    REPO/'src/matlab/metrics/compute_foodweb_metrics.m',
    REPO/'docs/stats/apply_wlnm_tukey_retention.py']
save(pd.DataFrame([dict(File=str(f), SHA256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in source_files]), 'source_hashes.csv')
print(summary.to_string(index=False))
print('Baseline independently reproduced for all 2610 foodweb/ratio groups; 130500 raw rows.')
print('Status counts:', counts.to_dict('records'))
