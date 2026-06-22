# Athen web-level equivalence analysis

Primary unit: one empirical food web compared with the mean of its pseudo reconstructions.
Relative error: (mean pseudo - empirical) / empirical.
TOST alpha: 0.05; equivalence uses the 90% CI and relative margins 10%, 15%, 20%, 30%.

Mean trophic height has two explicitly labelled populations:
ConditionalAtLeast10ValidRuns and Complete20Sensitivity.
The difference between them quantifies sensitivity to undefined trophic heights.
Trophic-height results remain provisional until the discrepancies with Lachlan's CSV are resolved.

Log files: 290
Raw rows: 52200
Main train ratio: 60

## Main train-ratio results

| Metric | Population | Margin | Mean relative error | 90% CI | Equivalent |
|---|---|---:|---:|---:|---:|
| Connectance | PrimaryMean20 | 10% | -15.20% | [-16.04%, -14.37%] | 0 |
| Connectance | PrimaryMean20 | 15% | -15.20% | [-16.04%, -14.37%] | 0 |
| Connectance | PrimaryMean20 | 20% | -15.20% | [-16.04%, -14.37%] | 1 |
| Connectance | PrimaryMean20 | 30% | -15.20% | [-16.04%, -14.37%] | 1 |
| MeanGenerality | PrimaryMean20 | 10% | -11.68% | [-12.59%, -10.76%] | 0 |
| MeanGenerality | PrimaryMean20 | 15% | -11.68% | [-12.59%, -10.76%] | 1 |
| MeanGenerality | PrimaryMean20 | 20% | -11.68% | [-12.59%, -10.76%] | 1 |
| MeanGenerality | PrimaryMean20 | 30% | -11.68% | [-12.59%, -10.76%] | 1 |
| MeanVulnerability | PrimaryMean20 | 10% | -10.44% | [-11.33%, -9.55%] | 0 |
| MeanVulnerability | PrimaryMean20 | 15% | -10.44% | [-11.33%, -9.55%] | 1 |
| MeanVulnerability | PrimaryMean20 | 20% | -10.44% | [-11.33%, -9.55%] | 1 |
| MeanVulnerability | PrimaryMean20 | 30% | -10.44% | [-11.33%, -9.55%] | 1 |
| MeanTrophicHeight | ConditionalAtLeast10ValidRuns | 10% | 12.12% | [9.55%, 14.68%] | 0 |
| MeanTrophicHeight | ConditionalAtLeast10ValidRuns | 15% | 12.12% | [9.55%, 14.68%] | 1 |
| MeanTrophicHeight | ConditionalAtLeast10ValidRuns | 20% | 12.12% | [9.55%, 14.68%] | 1 |
| MeanTrophicHeight | ConditionalAtLeast10ValidRuns | 30% | 12.12% | [9.55%, 14.68%] | 1 |
| MeanTrophicHeight | Complete20Sensitivity | 10% | 5.12% | [3.77%, 6.47%] | 1 |
| MeanTrophicHeight | Complete20Sensitivity | 15% | 5.12% | [3.77%, 6.47%] | 1 |
| MeanTrophicHeight | Complete20Sensitivity | 20% | 5.12% | [3.77%, 6.47%] | 1 |
| MeanTrophicHeight | Complete20Sensitivity | 30% | 5.12% | [3.77%, 6.47%] | 1 |
