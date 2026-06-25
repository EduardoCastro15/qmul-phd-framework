# Outlier-filtered web-level reconstruction summary, train ratio 67

This output is derived from the original WLNM logs; the original logs are not modified.
One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.
CvK: 3
Trophic-height source metric: MeanTrophicLevel.
Outliers are pseudo-run metric values outside Tukey fences: Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
Fences are calculated within each food web and metric.
Food-web metrics are included only when at least 10 pseudo-runs remain after filtering.

## Overall summary

| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Connectance | 290 | 301 | 0.1324 | 0.1158 | -0.0165 | -12.58% [-13.32%, -11.84%] |
| Mean trophic height | 290 | 1852 | 2.1841 | 2.2958 | +0.1117 | +4.60% [+3.31%, +5.89%] |
| Mean generality | 290 | 324 | 10.5377 | 9.1763 | -1.3614 | -10.90% [-11.67%, -10.14%] |
| Mean vulnerability | 290 | 289 | 8.2904 | 7.3362 | -0.9541 | -9.56% [-10.31%, -8.82%] |
