# Outlier-filtered web-level reconstruction summary, train ratio 60

This output is derived from the original WLNM logs; the original logs are not modified.
One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.
CvK: not used as grouping variable
ThresholdMode filter: threshold_sweep.
Classification-threshold filter: 0.5.
Trophic-height source metric: NetworkXMeanTrophicLevel.
Outliers are pseudo-run metric values outside Tukey fences: Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
Fences are calculated within each food web and metric.
Food-web metrics are included only when at least 0 pseudo-runs remain after filtering.

## Overall summary

| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Connectance | 290 | 266 | 0.1324 | 0.1139 | -0.0185 | -14.48% [-15.38%, -13.57%] |
| Mean trophic height | 290 | 1088 | 2.1898 | 2.3426 | +0.1528 | +5.30% [+3.99%, +6.61%] |
| Mean generality | 290 | 355 | 10.5418 | 8.9031 | -1.6388 | -11.67% [-12.62%, -10.72%] |
| Mean vulnerability | 290 | 316 | 8.2909 | 7.1690 | -1.1218 | -10.38% [-11.32%, -9.45%] |
