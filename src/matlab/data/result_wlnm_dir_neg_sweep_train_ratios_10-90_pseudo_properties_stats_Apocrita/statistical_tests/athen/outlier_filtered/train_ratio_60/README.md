# Outlier-filtered web-level reconstruction summary, train ratio 60

This output is derived from the original WLNM logs; the original logs are not modified.
One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.
Outliers are pseudo-run metric values outside Tukey fences: Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.
Fences are calculated within each food web and metric.
Food-web metrics are included only when at least 10 pseudo-runs remain after filtering.

## Overall summary

| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Connectance | 290 | 202 | 0.1324 | 0.1123 | -0.0201 | -15.40% [-16.25%, -14.54%] |
| Mean trophic height | 289 | 447 | 2.6442 | 2.7688 | +0.1246 | +3.76% [+2.94%, +4.59%] |
| Mean generality | 290 | 227 | 10.5418 | 8.9632 | -1.5786 | -12.09% [-13.02%, -11.17%] |
| Mean vulnerability | 290 | 178 | 8.2909 | 7.1809 | -1.1100 | -10.68% [-11.58%, -9.78%] |
