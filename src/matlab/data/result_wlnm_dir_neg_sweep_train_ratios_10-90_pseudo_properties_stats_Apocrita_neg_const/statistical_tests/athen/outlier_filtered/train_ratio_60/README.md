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
| Connectance | 290 | 296 | 0.1324 | 0.1190 | -0.0133 | -11.30% [-12.59%, -10.01%] |
| Mean trophic height | 290 | 913 | 2.1898 | 2.3471 | +0.1572 | +5.40% [+3.15%, +7.64%] |
| Mean generality | 290 | 350 | 10.5418 | 9.4286 | -1.1132 | -7.33% [-8.72%, -5.93%] |
| Mean vulnerability | 290 | 257 | 8.2909 | 7.5265 | -0.7644 | -7.33% [-8.59%, -6.07%] |
