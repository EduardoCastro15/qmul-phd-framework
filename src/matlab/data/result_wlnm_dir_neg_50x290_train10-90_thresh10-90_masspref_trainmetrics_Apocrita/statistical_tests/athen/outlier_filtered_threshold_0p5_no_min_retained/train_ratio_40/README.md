# Outlier-filtered web-level reconstruction summary, train ratio 40

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
| Connectance | 290 | 250 | 0.1324 | 0.1060 | -0.0264 | -20.87% [-22.01%, -19.74%] |
| Mean trophic height | 290 | 1175 | 2.1898 | 2.6551 | +0.4653 | +16.49% [+13.56%, +19.42%] |
| Mean generality | 290 | 371 | 10.5418 | 8.2270 | -2.3148 | -16.34% [-17.58%, -15.10%] |
| Mean vulnerability | 290 | 291 | 8.2909 | 6.7026 | -1.5883 | -14.39% [-15.67%, -13.11%] |
