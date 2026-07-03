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
| Connectance | 290 | 369 | 0.1324 | 0.1280 | -0.0044 | -3.54% [-3.84%, -3.23%] |
| Mean trophic height | 290 | 1967 | 2.1898 | 2.1823 | -0.0075 | -0.30% [-0.46%, -0.13%] |
| Mean generality | 290 | 628 | 10.5418 | 10.1717 | -0.3701 | -2.70% [-3.02%, -2.39%] |
| Mean vulnerability | 290 | 426 | 8.2909 | 8.0235 | -0.2673 | -2.62% [-2.91%, -2.33%] |
