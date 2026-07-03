# Outlier-filtered web-level reconstruction summary, train ratio 90

This output is derived from the original WLNM logs; the original logs are not modified.
One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.
CvK: not used as grouping variable
Trophic-height source metric: NetworkXMeanTrophicLevel.
Outliers are pseudo-run metric values outside Tukey fences: Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
Fences are calculated within each food web and metric.
Food-web metrics are included only when at least 10 pseudo-runs remain after filtering.

## Overall summary

| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Connectance | 207 | 314 | 0.1260 | 0.1209 | -0.0051 | -4.10% [-4.44%, -3.76%] |
| Mean trophic height | 207 | 1570 | 1.8643 | 1.8599 | -0.0045 | -0.23% [-0.35%, -0.12%] |
| Mean generality | 207 | 547 | 9.1146 | 8.7064 | -0.4083 | -3.34% [-3.70%, -2.99%] |
| Mean vulnerability | 207 | 352 | 6.4000 | 6.1519 | -0.2481 | -3.00% [-3.34%, -2.66%] |
