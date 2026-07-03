# Outlier-filtered web-level reconstruction summary, train ratio 60

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
| Connectance | 207 | 228 | 0.1260 | 0.1085 | -0.0175 | -14.61% [-15.78%, -13.44%] |
| Mean trophic height | 207 | 726 | 1.8643 | 1.9320 | +0.0676 | +2.78% [+0.73%, +4.84%] |
| Mean generality | 207 | 288 | 9.1146 | 7.6848 | -1.4299 | -11.21% [-12.45%, -9.96%] |
| Mean vulnerability | 207 | 202 | 6.4000 | 5.5622 | -0.8378 | -10.03% [-11.21%, -8.85%] |
