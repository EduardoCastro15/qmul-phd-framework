# Outlier-filtered web-level reconstruction summary, train ratio 90

This output is derived from the original WLNM logs; the original logs are not modified.
One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.
Outliers are pseudo-run metric values outside Tukey fences: Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
Fences are calculated within each food web and metric.
Food-web metrics are included only when at least 10 pseudo-runs remain after filtering.

## Overall summary

| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Connectance | 290 | 161 | 0.1324 | 0.1267 | -0.0057 | -4.20% [-4.46%, -3.95%] |
| Mean trophic height | 290 | 780 | 2.6445 | 2.6499 | +0.0054 | +0.10% [-0.04%, +0.25%] |
| Mean generality | 290 | 337 | 10.5418 | 10.1083 | -0.4335 | -3.52% [-3.78%, -3.26%] |
| Mean vulnerability | 290 | 246 | 8.2909 | 7.9849 | -0.3060 | -3.21% [-3.47%, -2.95%] |
