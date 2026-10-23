# SEAL_directed diagrams and edge-conditioned proposal

## Current implementation

The two `seal_directed_attrs_*` diagrams describe the implemented and validated
`SEAL_directed_gpu_v1` pipeline. Direction is retained in ordered candidate
pairs, target-edge masking, source/target node labels, and role-constrained
negative sampling. The present DGCNN backend receives a weak-undirected
projection of each subgraph. Its node input concatenates the structural
one-hot tag with 44 ecological attributes; Node2Vec and edge features are off.

The 44 node attributes are:

- 2 mass channels: standardized log10 mass and a missingness indicator;
- 6 taxonomy-level categories;
- 10 metabolic-type categories;
- 7 movement-type categories;
- 19 life-stage channels.

The `role` vector constrains negative candidates but is not part of the learned
44-column node feature matrix.

## Recommended paper

**Simonovsky, M. & Komodakis, N. (2017). Dynamic Edge-Conditioned Filters in
Convolutional Neural Networks on Graphs. CVPR.**

- Paper: https://arxiv.org/abs/1704.02901
- DOI landing page: https://doi.org/10.48550/arXiv.1704.02901

This is a strong methodological match because Edge-Conditioned Convolution
(ECC) makes each neighborhood filter a learned function of the corresponding
edge label or covariate. That directly supports a SEAL extension in which
observed context edges carry continuous and categorical ecological information.
The official SEAL_OGB implementation also provides a useful engineering
precedent for transporting `edge_weight` through the SEAL pipeline:
https://github.com/facebookresearch/SEAL_OGB/blob/main/seal_link_pred.py

## Proposed edge channels

The safest first candidate is:

`log10(consumer mass / resource mass)` plus a mass-missing indicator.

This is a **derived relational edge covariate**, not an observed interaction
strength. Because the two masses already exist among the node attributes, the
channel introduces a useful relational inductive bias but not entirely new raw
information. It must therefore be tested as an ablation, not assumed to improve
performance.

Possible later channels, subject to coverage and leakage checks, are one-hot
interaction type, interaction dimensionality, and interaction classification.
An actual interaction strength (biomass flux, feeding rate, interaction
frequency, or energy flow) should only be called a link weight when independently
measured and available under the intended inference setting.

## Leakage-safe implementation rules

1. Remove the target edge and every attribute belonging to that edge before
   extracting both positive and negative candidate subgraphs.
2. Attach edge covariates only to observed context edges.
3. Fit imputers and scalers using training data only; include missingness flags.
4. Keep splits, seeds, negative samples, training budget, and evaluation policy
   identical across ablations.
5. Distinguish predicting link existence from predicting interaction strength.

## Recommended ablation ladder

- A: binary structure only;
- B: A + current 44 node attributes;
- C: B + log mass-ratio edge channel;
- D: C + validated interaction descriptors.

The two `seal_directed_edge_weight_proposal_*` diagrams are explicitly marked
as proposals and do not describe code that is currently implemented.
