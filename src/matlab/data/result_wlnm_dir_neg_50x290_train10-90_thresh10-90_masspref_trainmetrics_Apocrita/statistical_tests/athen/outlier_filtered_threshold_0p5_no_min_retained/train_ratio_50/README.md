# Outlier-filtered web-level reconstruction summary, train ratio 50

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
| Connectance | 290 | 267 | 0.1324 | 0.1102 | -0.0222 | -17.47% [-18.51%, -16.43%] |
| Mean trophic height | 290 | 1111 | 2.1898 | 2.4605 | +0.2707 | +9.45% [+7.73%, +11.17%] |
| Mean generality | 290 | 291 | 10.5418 | 8.5617 | -1.9801 | -13.87% [-14.97%, -12.77%] |
| Mean vulnerability | 290 | 296 | 8.2909 | 6.9303 | -1.3606 | -12.31% [-13.43%, -11.19%] |
