# Outlier-filtered web-level reconstruction summary, train ratio 90

This output is derived from the original WLNM logs; the original logs are not modified.
One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.
CvK: 10
Trophic-height source metric: MeanTrophicLevel.
Outliers are pseudo-run metric values outside Tukey fences: Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
Fences are calculated within each food web and metric.
Food-web metrics are included only when at least 10 pseudo-runs remain after filtering.

## Overall summary

| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Connectance | 290 | 1090 | 0.1324 | 0.1269 | -0.0055 | -4.05% [-4.29%, -3.81%] |
| Mean trophic height | 290 | 9809 | 2.1841 | 2.1904 | +0.0064 | +0.14% [-0.03%, +0.31%] |
| Mean generality | 290 | 2698 | 10.5377 | 10.1032 | -0.4345 | -3.57% [-3.82%, -3.32%] |
| Mean vulnerability | 290 | 1967 | 8.2904 | 7.9858 | -0.3046 | -3.19% [-3.43%, -2.95%] |
