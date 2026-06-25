# Outlier-filtered web-level reconstruction summary, train ratio 80

This output is derived from the original WLNM logs; the original logs are not modified.
One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.
CvK: 5
Trophic-height source metric: MeanTrophicLevel.
Outliers are pseudo-run metric values outside Tukey fences: Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
Fences are calculated within each food web and metric.
Food-web metrics are included only when at least 10 pseudo-runs remain after filtering.

## Overall summary

| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Connectance | 290 | 646 | 0.1324 | 0.1219 | -0.0105 | -7.82% [-8.29%, -7.35%] |
| Mean trophic height | 290 | 3707 | 2.1841 | 2.2234 | +0.0393 | +1.51% [+0.93%, +2.10%] |
| Mean generality | 290 | 1004 | 10.5377 | 9.6956 | -0.8421 | -6.86% [-7.34%, -6.38%] |
| Mean vulnerability | 290 | 869 | 8.2904 | 7.6949 | -0.5955 | -6.11% [-6.56%, -5.65%] |
