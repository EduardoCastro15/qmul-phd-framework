# Ledger sanity comparison

This is a descriptive sanity comparison, not an equivalence test.
One observation is one empirical food web compared with the mean of its valid pseudo reconstructions.
Connectance is recalculated as L/S^2. NetworkX trophic height includes basal species at level 1 and is averaged over the largest weakly connected component.

The conservative correspondence rule requires the entire 90% confidence interval of the mean web-level relative error to be contained within +/- the relative SEM reported for the four Ledger control webs.
SEM describes precision of the four-web mean; it is not the full range of natural variation.

## TrainRatio 60 overall

| Metric | N | Mean empirical | Mean pseudo | Mean delta | Relative error (90% CI) | Ledger relative SEM | Corresponds? |
|---|---:|---:|---:|---:|---:|---:|---:|
| Connectance | 290 | 0.1283 | 0.1090 | -0.0193 | -15.19% [-16.03%, -14.35%] | +/-11.11% | 0 |
| Mean generality | 290 | 10.5418 | 9.0060 | -1.5358 | -11.68% [-12.59%, -10.76%] | +/-9.50% | 0 |
| Mean vulnerability | 290 | 8.2909 | 7.1990 | -1.0919 | -10.43% [-11.32%, -9.54%] | +/-9.50% | 0 |
| Mean trophic height | 290 | 2.1898 | 2.5425 | +0.3526 | +11.56% [+8.99%, +14.12%] | +/-0.26% | 0 |
