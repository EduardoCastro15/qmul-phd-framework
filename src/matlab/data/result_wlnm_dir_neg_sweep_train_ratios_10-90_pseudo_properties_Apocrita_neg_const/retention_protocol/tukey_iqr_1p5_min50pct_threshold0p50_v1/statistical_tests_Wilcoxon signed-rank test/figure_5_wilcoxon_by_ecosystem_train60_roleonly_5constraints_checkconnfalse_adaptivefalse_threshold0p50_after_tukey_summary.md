# Figure 5 Wilcoxon signed-rank tests by ecosystem type

- Train ratio: 60%
- Threshold: 0.50
- Unit of analysis: food web
- Difference: post-Tukey pseudo mean minus empirical value
- Test: paired, two-sided Wilcoxon signed-rank; `zero_method=wilcox`; `method=auto`
- Primary multiplicity correction: Holm across all 20 ecosystem-by-metric tests
- Protocol-matched correction: Holm within each ecosystem across four ecological metrics
- Additional correction: Holm within each metric across five ecosystem types

A positive median difference means that pseudo food webs have a higher value; a negative value means they have a lower value.

| Ecosystem | Metric | n | Median difference | Rank-biserial r | Raw p | Holm p (20) | Reject global H0 |
|---|---|---:|---:|---:|---:|---:|:---:|
| Lakes | Connectance | 55 | -0.00805975 | -0.930 | 1.98e-09 | 3.57e-08 | Yes |
| Lakes | Mean trophic height | 55 | 0.0192245 | 0.483 | 0.0018 | 0.0128 | Yes |
| Lakes | Mean generality | 55 | -0.376038 | -0.718 | 3.60e-06 | 4.32e-05 | Yes |
| Lakes | Mean vulnerability | 55 | -0.120415 | -0.801 | 2.35e-07 | 3.52e-06 | Yes |
| Streams | Connectance | 28 | -0.0212241 | -0.941 | 5.22e-07 | 7.30e-06 | Yes |
| Streams | Mean trophic height | 28 | 0.0124321 | 0.473 | 0.0281 | 0.1122 | No |
| Streams | Mean generality | 28 | -3.88149 | -0.995 | 1.49e-08 | 2.38e-07 | Yes |
| Streams | Mean vulnerability | 28 | -1.63975 | -0.921 | 1.26e-06 | 1.64e-05 | Yes |
| Marine | Connectance | 134 | -0.0123352 | -0.792 | 1.74e-15 | 3.49e-14 | Yes |
| Marine | Mean trophic height | 133 | -0.00944464 | 0.027 | 0.7867 | 1.0000 | No |
| Marine | Mean generality | 134 | -0.251944 | -0.587 | 3.69e-09 | 6.28e-08 | Yes |
| Marine | Mean vulnerability | 134 | -0.299073 | -0.618 | 5.51e-10 | 1.05e-08 | Yes |
| Terrestrial aboveground | Connectance | 21 | -0.0157118 | -0.896 | 6.68e-05 | 7.34e-04 | Yes |
| Terrestrial aboveground | Mean trophic height | 21 | 0.164935 | 0.861 | 1.61e-04 | 0.0016 | Yes |
| Terrestrial aboveground | Mean generality | 21 | -2.66287 | -0.827 | 3.54e-04 | 0.0032 | Yes |
| Terrestrial aboveground | Mean vulnerability | 21 | -1.32243 | -0.818 | 4.26e-04 | 0.0034 | Yes |
| Terrestrial belowground | Connectance | 52 | -0.0142768 | -0.450 | 0.0048 | 0.0285 | Yes |
| Terrestrial belowground | Mean trophic height | 51 | -0.176592 | -0.077 | 0.6326 | 1.0000 | No |
| Terrestrial belowground | Mean generality | 52 | -0.790385 | -0.295 | 0.0645 | 0.1935 | No |
| Terrestrial belowground | Mean vulnerability | 52 | -1.37502 | -0.367 | 0.0212 | 0.1061 | No |

## Plain-language summary

After the global Holm correction, 15 of 20 ecosystem-by-metric comparisons reject the null hypothesis of a zero median paired difference.

Globally significant comparisons:

- Lakes, Connectance: pseudo values are typically lower (median difference -0.00805975, Holm p=3.57e-08, rank-biserial r=-0.930).
- Lakes, Mean trophic height: pseudo values are typically higher (median difference 0.0192245, Holm p=0.0128, rank-biserial r=0.483).
- Lakes, Mean generality: pseudo values are typically lower (median difference -0.376038, Holm p=4.32e-05, rank-biserial r=-0.718).
- Lakes, Mean vulnerability: pseudo values are typically lower (median difference -0.120415, Holm p=3.52e-06, rank-biserial r=-0.801).
- Streams, Connectance: pseudo values are typically lower (median difference -0.0212241, Holm p=7.30e-06, rank-biserial r=-0.941).
- Streams, Mean generality: pseudo values are typically lower (median difference -3.88149, Holm p=2.38e-07, rank-biserial r=-0.995).
- Streams, Mean vulnerability: pseudo values are typically lower (median difference -1.63975, Holm p=1.64e-05, rank-biserial r=-0.921).
- Marine, Connectance: pseudo values are typically lower (median difference -0.0123352, Holm p=3.49e-14, rank-biserial r=-0.792).
- Marine, Mean generality: pseudo values are typically lower (median difference -0.251944, Holm p=6.28e-08, rank-biserial r=-0.587).
- Marine, Mean vulnerability: pseudo values are typically lower (median difference -0.299073, Holm p=1.05e-08, rank-biserial r=-0.618).
- Terrestrial aboveground, Connectance: pseudo values are typically lower (median difference -0.0157118, Holm p=7.34e-04, rank-biserial r=-0.896).
- Terrestrial aboveground, Mean trophic height: pseudo values are typically higher (median difference 0.164935, Holm p=0.0016, rank-biserial r=0.861).
- Terrestrial aboveground, Mean generality: pseudo values are typically lower (median difference -2.66287, Holm p=0.0032, rank-biserial r=-0.827).
- Terrestrial aboveground, Mean vulnerability: pseudo values are typically lower (median difference -1.32243, Holm p=0.0034, rank-biserial r=-0.818).
- Terrestrial belowground, Connectance: pseudo values are typically lower (median difference -0.0142768, Holm p=0.0285, rank-biserial r=-0.450).

A non-significant result does not demonstrate equivalence or similarity; it only means that this test did not provide sufficient evidence of a non-zero paired difference for that ecosystem and metric.
