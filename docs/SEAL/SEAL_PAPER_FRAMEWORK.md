# SEAL paper framework

## 1. Scientific objective and positioning

**Provisional title:** *Transferable direction-aware subgraph learning reveals
when trophic interactions can be predicted across ecosystems.*

**Central question:** Can a fully directed SEAL model learn trophic assembly
rules from local structure and species traits that generalise to unseen food
webs and ecosystems, while maintaining calibrated predictions under incomplete
and biased observations?

The intended contribution combines ecology and methodology. It is not simply
an application of SEAL to food webs and it is not a model leaderboard. The
paper must establish both:

1. a technical advance in directed, transferable, uncertainty-aware subgraph
   learning; and
2. an ecological result explaining when structure, direction, traits, or their
   interaction make trophic links predictable.

### Boundary with the WLNM paper

- **WLNM:** within-network reconstruction using matrix encoding and ecological
  negative sampling.
- **SEAL:** cross-network inductive learning, genuinely directed message
  passing, traits, positive--unlabelled learning, calibration, and temporal
  external validation.
- The papers must not share a headline comparison, principal figure, or central
  conclusion.

### Editorial ladder

1. **Nature Ecology & Evolution** only if the work produces external
   generalisation and a broad ecological conclusion.
2. **Nature Communications** if the method and external evidence are strong but
   the ecological conceptual advance is narrower.
3. **Methods in Ecology and Evolution** if the main outcome is a robust,
   reusable ecological method.

The target will be selected at the Week 20 gate from the evidence, not from a
preference for journal prestige.

## 2. Verified starting point

The repository currently contains:

- 290 food webs and 18,721 nodes with attributed MAT files;
- 870 directed-SEAL runs, corresponding to three seeds per food web;
- 44 node features covering mass, taxonomic resolution, metabolic type,
  movement type, life stage, and missingness indicators;
- preliminary improvements over the original SEAL results, particularly in
  precision; and
- a directed candidate and masking pipeline whose DGCNN backend still performs
  message passing on the weak undirected projection of each subgraph.

The existing result comparison is exploratory because it simultaneously
changes direction handling, node attributes, negative-link sampling, repeats,
and some realised train ratios. It cannot identify the causal contribution of
any component and must not be used as the paper's primary evidence.

### Novelty constraint

Recent work already covers:

- [inductive prediction across ecological networks](https://www.nature.com/articles/s41559-025-02715-6);
- [direction, structure, and traits on the same 290 food webs using stacked generalisation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12976110/);
- [evaluation guidance for species-interaction prediction](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14071); and
- [bias-aware ecological link prediction](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70368).

Therefore, adding traits to SEAL or reporting higher within-network accuracy is
not sufficient novelty. The paper must demonstrate transferable directed
learning, uncertainty-aware treatment of non-observations, and an independent
ecological validation.

## 3. Technical framework

### 3.1 Implementation strategy

A new implementation will be developed in `src/python/eco_seal/` using PyTorch
Geometric. The current code in `src/python/seal_directed/` remains frozen as a
legacy baseline and provenance record.

The final model will include:

- directed enclosing-subgraph extraction;
- removal of only the candidate `resource -> consumer` edge;
- preservation of an observed reverse edge as context;
- distinct resource and consumer endpoint labels;
- forward and reverse relation types in message passing;
- an optional trait encoder with explicit missingness masks;
- graph pooling and an edge classifier;
- positive--unlabelled training; and
- validation-only probability calibration and uncertainty reporting.

### 3.2 Data contracts

Three versioned objects will define the public research interface.

#### `GraphRecord`

- `network_id`
- source dataset and provenance
- ecosystem and temporal identifiers
- node identifiers
- directed `resource -> consumer` edge index
- node traits and missingness masks
- sampling-effort fields when available

#### `EdgeExample`

- resource node
- consumer node
- state: `positive` or `unlabelled`
- split and fold identifiers
- candidate-generation protocol

#### `SplitManifest`

- hashes of source and derived datasets
- random seed and protocol version
- train, validation, and test networks and edges
- feature-fitting scope
- leakage and overlap checks

Planned command-line entry points:

```text
python -m eco_seal.prepare  --config <yaml>
python -m eco_seal.train    --config <yaml>
python -m eco_seal.evaluate --run-dir <path>
python -m eco_seal.report   --registry <path>
```

Every run must record the resolved configuration, commit, dependency versions,
seed, split hash, metrics, checkpoint, runtime, and terminal log.

### 3.3 Baselines and ablations

All methods must consume identical split manifests and candidate sets.

1. Directed structural heuristics and an ecological trait rule.
2. Trait-only MLP.
3. Frozen legacy SEAL/DGCNN.
4. PyG-SEAL using an undirected projection, as a parity baseline.
5. Fully directed SEAL using structure only.
6. Fully directed SEAL with traits.
7. Fully directed SEAL with traits, positive--unlabelled learning, and
   calibration.
8. The published stacked-generalisation method on matched splits.
9. BUDDY on an undirected projection as a modern generic link-prediction
   baseline.

Ecological role filtering will be a sensitivity analysis rather than the
definition of a true negative. Roles, topology-derived features, imputations,
and scalers must be fitted using training data only.

### 3.4 Evaluation protocols

#### P1. Network-disjoint evaluation -- primary

Five folds of complete networks, stratified by ecosystem, size, and
connectance. No labelled link from a test network may be used to train the
model.

#### P2. Ecosystem transfer

Leave-one-ecosystem-out evaluation across the five established ecosystem
groups.

#### P3. Within-network transductive benchmark

Five edge folds per food web for comparison with prior work. This is secondary
to network-disjoint generalisation.

#### P4. Controlled missingness

Remove 10%, 30%, and 50% of observed links under:

- missing completely at random;
- degree-biased observation;
- trait-biased observation; and
- sampling-effort-biased observation where effort is available.

#### P5. DAPSTOM temporal validation

Use rolling-origin evaluation by sea and decade. Train only on information
available through decade `t` and predict links first observed in `t+1`.
Unobserved pairs remain unlabelled. Warm-start predictions form the primary
analysis; cold-start taxa are reported separately.

### 3.5 Metrics and inference

Primary metrics:

- Average Precision relative to prevalence;
- Precision@K and Recall@K, with both link-count and fixed-effort budgets;
- Brier score;
- Expected Calibration Error; and
- coverage versus risk for models allowed to abstain.

ROC-AUC is secondary and will not support the principal claim.

Repetition policy:

- three seeds for pilots;
- five seeds for the full experiment matrix; and
- ten seeds for final models and principal baselines.

The food web is the inferential unit. Seed-level values will first be
aggregated within web. Principal contrasts will report paired effect sizes and
network-clustered bootstrap confidence intervals. Secondary hypothesis tests
will use Holm correction. No post-hoc outlier deletion will be applied without
a predeclared data-quality rule and a complete sensitivity analysis.

## 4. Weekly operating system

The weekly 15--20 hour allocation is:

- 2 hours: literature radar and one deep reading;
- 8--10 hours: data, implementation, or experiments;
- 3 hours: analysis and visualisation;
- 2 hours: cumulative manuscript writing; and
- 1 hour: decision log, reproducibility, and next-sprint planning.

A week is complete only when it produces:

1. one verifiable primary artefact;
2. one diagnostic table or figure;
3. one reusable Methods or Results paragraph;
4. updated evidence and decision logs; and
5. a concrete first action for the following week.

## 5. Twenty-six-week roadmap

| Week | Dates | Primary deliverable |
|---|---|---|
| 1 | 28 Sep--4 Oct 2026 | Paper charter, scope boundary, provisional CRediT statement, and written confirmation of leadership. |
| 2 | 5--11 Oct | Novelty matrix covering SEAL, BUDDY, NCNC, ecological ILP, stacked models, and COIL+. |
| 3 | 12--18 Oct | Leakage and reproducibility audit of current data, traits, roles, seeds, and result files. |
| 4 | 19--25 Oct | Common Gateway--DAPSTOM schema and immutable split manifests. **Gate 1: data validity.** |
| 5 | 26 Oct--1 Nov | Reproducible PyG environment and Apocrita GPU time/memory benchmark. |
| 6 | 2--8 Nov | Legacy DGCNN reproduction on 20 stratified sentinel food webs. |
| 7 | 9--15 Nov | Undirected PyG-SEAL parity implementation. |
| 8 | 16--22 Nov | Directed enclosing subgraphs and directed message passing. |
| 9 | 23--29 Nov | Trait encoder with train-only fitting and missingness masks. |
| 10 | 30 Nov--6 Dec | Positive--unlabelled objective and role-filter sensitivity. |
| 11 | 7--13 Dec | Calibration, reliability diagrams, and abstention policy. |
| 12 | 14--20 Dec | Three-seed factorial pilot on 20 food webs. **Gate 2: architecture selection.** |
| 13 | 21--27 Dec | Refactoring, documentation, literature synthesis, and technical-debt closure. |
| 14 | 28 Dec--3 Jan 2027 | Missingness simulator and recovery tests. |
| 15 | 4--10 Jan | Full within-network matched benchmark. |
| 16 | 11--17 Jan | Five-fold network-disjoint experiment. |
| 17 | 18--24 Jan | Leave-one-ecosystem-out transfer matrix. |
| 18 | 25--31 Jan | DAPSTOM adapter and sea--decade temporal sequences. |
| 19 | 1--7 Feb | DAPSTOM rolling-origin evaluation. |
| 20 | 8--14 Feb | Principal statistics, calibration, and robustness. **Gate 3: journal selection.** |
| 21 | 15--21 Feb | Trait-family permutations, structural ablations, and directed-motif interpretation. |
| 22 | 22--28 Feb | Complete first version of all principal figures and the results narrative. |
| 23 | 1--7 Mar | Methods, Data, Results, and automated supplementary tables. |
| 24 | 8--14 Mar | Introduction and Discussion centred on ecological transfer and limitations. |
| 25 | 15--21 Mar | Clean-environment reproduction and internal manuscript/code audit. |
| 26 | 22--28 Mar | Frozen manuscript and preprint, public release, and editorial enquiry. |

## 6. Literature workflow

Each reviewed paper must receive one row in
`literature/literature_matrix.csv` containing:

- research question and claimed contribution;
- datasets and inferential unit;
- split protocol;
- negative or unlabelled construction;
- leakage risks;
- metrics and statistical design;
- code and data availability;
- relevance to the SEAL paper; and
- a concrete adopt, test, or reject decision.

Every four weeks, the weekly tracker will include a one-page synthesis of what
the literature changed in the project design.

## 7. Figure programme

1. **Method:** incomplete directed food web to attributed subgraphs, directed
   encoder, cross-network transfer, and calibrated ranking.
2. **Causal ablation:** paired effects of direction, traits, and
   positive--unlabelled learning.
3. **Generalisation:** ecosystem transfer matrix and degradation under
   controlled missingness.
4. **Ecological interpretation:** contribution of motifs and trait families
   versus size, connectance, trait coverage, and ecosystem.
5. **DAPSTOM:** temporal prediction, calibration, and priority links under
   fixed sampling budgets.

All principal figures will be script-generated in vector PDF/SVG plus a
high-resolution raster copy. They will use an accessible palette, display
uncertainty and inferential units, avoid three-dimensional decoration, and
state `n`, aggregation, and metric definitions in the caption.

## 8. Authorship and open-science protocol

The existing verbal first-authorship agreement will be converted during Week 1
into a written paper charter. The provisional lead-author contribution is:

- Conceptualization
- Methodology
- Software
- Data curation
- Formal analysis
- Validation
- Visualization
- Project administration
- Writing -- original draft

Supervision, Resources, and Writing -- review & editing will be assigned from
actual contributions. The CRediT matrix and provisional order will be reviewed
monthly. Autonomy and documented leadership support first authorship, but all
authors must explicitly agree to the final author list, order, and contribution
statement.

The public release will contain:

- source code and pinned environment;
- split manifests and hashes;
- resolved experiment configurations;
- final model weights; and
- derived results and figure-generation scripts.

Raw data will be linked or redistributed only as permitted by the Gateway and
DAPSTOM licences.

## 9. Acceptance criteria and editorial gates

The study is scientifically complete when:

- every comparison uses identical splits and eligible candidates;
- no test information enters roles, features, normalisation, candidate
  generation, model selection, or calibration;
- the principal result survives network-disjoint and ecosystem-transfer tests;
- DAPSTOM demonstrates temporal utility rather than only random edge recovery;
- calibration and important failure modes are reported;
- direction, traits, and positive--unlabelled learning are separated by
  controlled ablations; and
- a clean environment reproduces at least one principal figure end to end.

Nature Ecology & Evolution will be considered only if Week 20 shows all of:

1. consistent improvement over matched specialist and generic baselines;
2. transfer to unseen networks and ecosystems;
3. independent temporal validation;
4. a broad ecological conclusion about food-web predictability; and
5. a reusable open model and workflow.

If criterion 4 is absent, the default target is Nature Communications. If the
advance is predominantly methodological, the default target is Methods in
Ecology and Evolution.

## 10. Fixed assumptions

- Project period: 28 September 2026 to 28 March 2027.
- Effort: 15--20 hours per week.
- Compute: Apocrita GPU.
- External validation: DAPSTOM sea--decade networks.
- Implementation: new PyTorch Geometric package with frozen legacy baseline.
- Release: code, manifests, configurations, and final model weights.
- `eco_seal` and the provisional paper title may be renamed after the Week 2
  novelty review without changing the scientific scope.

